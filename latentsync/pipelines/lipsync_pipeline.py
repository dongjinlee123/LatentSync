# Adapted from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/pipelines/pipeline_animation.py

import inspect
import os
import shutil
from typing import Callable, List, Optional, Union
import subprocess

import numpy as np
import torch
import torchvision

from diffusers.utils import is_accelerate_available
from packaging import version

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging

from einops import rearrange
import cv2

from ..models.unet import UNet3DConditionModel
from ..utils.image_processor import ImageProcessor
from ..utils.util import read_video, read_audio, write_video, check_ffmpeg_installed
from ..whisper.audio2feature import Audio2Feature
import tqdm
import soundfile as sf
import time
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class LipsyncPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        audio_encoder: Audio2Feature,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.set_progress_bar_config(desc="Steps")

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def decode_latents(self, latents):
        latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        decoded_latents = self.vae.decode(latents).sample
        return decoded_latents

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, height, width, callback_steps):
        assert height == width, "Height and width must be equal"

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_frames, num_channels_latents, height, width, dtype, device, generator):
        shape = (
            batch_size,
            num_channels_latents,
            1,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        rand_device = "cpu" if device.type == "mps" else device
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        latents = latents.repeat(1, 1, num_frames, 1, 1)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_mask_latents(
        self, mask, masked_image, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        masked_image = masked_image.to(device=device, dtype=dtype)

        # encode the mask image into latents space so we can concatenate it to the latents
        masked_image_latents = self.vae.encode(masked_image).latent_dist.sample(generator=generator)
        masked_image_latents = (masked_image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        mask = mask.to(device=device, dtype=dtype)

        # assume batch size = 1
        mask = rearrange(mask, "f c h w -> 1 c f h w")
        masked_image_latents = rearrange(masked_image_latents, "f c h w -> 1 c f h w")

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )
        return mask, masked_image_latents

    def prepare_image_latents(self, images, device, dtype, generator, do_classifier_free_guidance):
        images = images.to(device=device, dtype=dtype)
        image_latents = self.vae.encode(images).latent_dist.sample(generator=generator)
        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        image_latents = rearrange(image_latents, "f c h w -> 1 c f h w")
        image_latents = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents

        return image_latents

    def set_progress_bar_config(self, **kwargs):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(kwargs)

    @staticmethod
    def paste_surrounding_pixels_back(decoded_latents, pixel_values, masks, device, weight_dtype):
        # Paste the surrounding pixels back, because we only want to change the mouth region
        pixel_values = pixel_values.to(device=device, dtype=weight_dtype)
        masks = masks.to(device=device, dtype=weight_dtype)
        combined_pixel_values = decoded_latents * masks + pixel_values * (1 - masks)
        return combined_pixel_values

    @staticmethod
    def pixel_values_to_images(pixel_values: torch.Tensor):
        pixel_values = rearrange(pixel_values, "f c h w -> f h w c")
        pixel_values = (pixel_values / 2 + 0.5).clamp(0, 1)
        images = (pixel_values * 255).to(torch.uint8)
        images = images.cpu().numpy()
        return images

    def affine_transform_video(self, video_path, audio_duration=None, video_fps=25):
        """
        Transform video frames and extend if needed to match audio duration, even when loading from cache.
        Args:
            video_path: Path to the video file
            audio_duration: Duration of audio in seconds
            video_fps: Video frames per second
        """
        # Create cache directory if it doesn't exist
        cache_dir = os.path.join(os.path.dirname(video_path), "face_alignment_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Generate base cache filename without audio duration
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        cache_file = os.path.join(cache_dir, f"{video_name}_alignment.npz")
        
        # Calculate required number of frames based on audio duration
        required_frames = int(audio_duration * video_fps) if audio_duration is not None else None
        
        if os.path.exists(cache_file):
            print(f"Loading cached face alignment from {cache_file}")
            cached_data = np.load(cache_file, allow_pickle=True)
            faces = torch.from_numpy(cached_data['faces'])
            video_frames = cached_data['video_frames']
            boxes = list(cached_data['boxes'])  # Convert to list for easier manipulation
            affine_matrices = list(cached_data['affine_matrices'])  # Convert to list for easier manipulation
            
            # Adjust cached data to match audio duration
            if required_frames is not None:
                current_frames = len(faces)
                if current_frames < required_frames:
                    # Extend by looping if too short
                    num_loops = (required_frames // current_frames) + 1
                    faces = faces.repeat(num_loops, 1, 1, 1)[:required_frames]
                    video_frames = np.tile(video_frames, (num_loops, 1, 1, 1))[:required_frames]
                    
                    # Extend boxes and affine_matrices by repeating
                    boxes_extended = []
                    affine_matrices_extended = []
                    for i in range(required_frames):
                        idx = i % current_frames
                        boxes_extended.append(boxes[idx])
                        affine_matrices_extended.append(affine_matrices[idx])
                    boxes = boxes_extended
                    affine_matrices = affine_matrices_extended
                    
                elif current_frames > required_frames:
                    # Trim if too long
                    faces = faces[:required_frames]
                    video_frames = video_frames[:required_frames]
                    boxes = boxes[:required_frames]
                    affine_matrices = affine_matrices[:required_frames]
                
            return faces, video_frames, boxes, affine_matrices
        
        # If no cache exists, perform face alignment
        video_frames = read_video(video_path, use_decord=False)
        
        # Adjust video frames length before processing if needed
        original_length = len(video_frames)
        if required_frames is not None and original_length < required_frames:
            num_loops = (required_frames // original_length) + 1
            video_frames = np.tile(video_frames, (num_loops, 1, 1, 1))[:required_frames]
        
        faces = []
        boxes = []
        affine_matrices = []
        print(f"Performing face alignment on {len(video_frames)} frames...")
        for frame in tqdm.tqdm(video_frames):
            face, box, affine_matrix = self.image_processor.affine_transform(frame)
            faces.append(face)
            boxes.append(box)
            affine_matrices.append(affine_matrix)

        faces = torch.stack(faces)
        
        # Save original (non-adjusted) results to cache
        print(f"Saving face alignment cache to {cache_file}")
        np.savez(
            cache_file,
            faces=faces.cpu().numpy(),
            video_frames=video_frames,
            boxes=boxes,
            affine_matrices=affine_matrices
        )
        
        # Adjust lengths if needed before returning
        if required_frames is not None:
            if len(faces) > required_frames:
                faces = faces[:required_frames]
                video_frames = video_frames[:required_frames]
                boxes = boxes[:required_frames]
                affine_matrices = affine_matrices[:required_frames]
        
        return faces, video_frames, boxes, affine_matrices
    
    def restore_video(self, faces, video_frames, boxes, affine_matrices):
        from concurrent.futures import ThreadPoolExecutor
        import multiprocessing

        video_frames = video_frames[: faces.shape[0]]
        batch_size = 32  # Process multiple frames at once
        out_frames = [None] * len(faces)  # Pre-allocate list with correct size
        
        # Determine number of workers based on CPU cores
        num_workers = min(multiprocessing.cpu_count(), 8)
        print(f"Restoring {len(faces)} faces using {num_workers} workers...")
        
        # Pre-compute all heights and widths
        heights = [int(box[3] - box[1]) for box in boxes]
        widths = [int(box[2] - box[0]) for box in boxes]
        
        def process_face(face, height, width):
            """Process a single face tensor into numpy array"""
            face = torchvision.transforms.functional.resize(face, size=(height, width), antialias=True)
            face = rearrange(face, "c h w -> h w c")
            face = (face / 2 + 0.5).clamp(0, 1)
            return (face * 255).to(torch.uint8).cpu().numpy()
        
        def restore_frame(args):
            """Restore a single frame with its processed face"""
            idx, face, frame, matrix = args
            face = process_face(face, heights[idx], widths[idx])
            restored = self.image_processor.restorer.restore_img(frame, face, matrix)
            return idx, restored
        
        # Process in batches to manage memory
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for i in tqdm.tqdm(range(0, len(faces), batch_size)):
                batch_slice = slice(i, min(i + batch_size, len(faces)))
                
                # Prepare batch arguments
                batch_args = [
                    (idx, faces[idx], video_frames[idx], affine_matrices[idx])
                    for idx in range(i, min(i + batch_size, len(faces)))
                ]
                
                # Process batch in parallel
                for idx, restored_frame in executor.map(restore_frame, batch_args):
                    out_frames[idx] = restored_frame
                
                # Optional: clear GPU memory after each batch
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
        
        return np.stack(out_frames, axis=0)

    @torch.no_grad()
    def __call__(
        self,
        video_path: str,
        audio_path: str,
        video_out_path: str,
        video_mask_path: str = None,
        num_frames: int = 16,
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.5,
        weight_dtype: Optional[torch.dtype] = torch.float16,
        eta: float = 0.0,
        mask: str = "fix_mask",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        is_train = self.unet.training
        self.unet.eval()

        check_ffmpeg_installed()
        start = time.time()
        # 0. Define call parameters
        batch_size = 1
        device = self._execution_device
        self.image_processor = ImageProcessor(height, mask=mask, device="cuda")
        self.set_progress_bar_config(desc=f"Sample frames: {num_frames}")

        
        audio_samples = read_audio(audio_path)
        audio_duration = len(audio_samples) / audio_sample_rate
        faces, original_video_frames, boxes, affine_matrices = self.affine_transform_video(
            video_path, 
            audio_duration=audio_duration,
            video_fps=video_fps
        )
        
        step_0 = time.time()
        # 1. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        step_1 = time.time()

        # 2. Check inputs
        self.check_inputs(height, width, callback_steps)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        step_2= time.time()

        # 3. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        step_3 = time.time()

        # 4. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        self.video_fps = video_fps

        if self.unet.add_audio_layer:
            whisper_feature = self.audio_encoder.audio2feat(audio_path)
            whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)

            num_inferences = min(len(faces), len(whisper_chunks)) // num_frames
        else:
            num_inferences = len(faces) // num_frames

        synced_video_frames = []
        masked_video_frames = []

        num_channels_latents = self.vae.config.latent_channels

        # Prepare latent variables
        all_latents = self.prepare_latents(
            batch_size,
            num_frames * num_inferences,
            num_channels_latents,
            height,
            width,
            weight_dtype,
            device,
            generator,
        )
        step_4 = time.time()
        inference_times = {
            'audio_processing': [],
            'prepare_masks': [],
            'prepare_mask_latents': [],
            'prepare_image_latents': [],
            'denoising_steps': [],
            'decode_latents': [],
            'paste_pixels': []
        }
        for i in tqdm.tqdm(range(num_inferences), desc="Doing inference..."):
            iter_start = time.time()
            
            # 1. Audio embedding preparation
            audio_start = time.time()
            if self.unet.add_audio_layer:
                audio_embeds = torch.stack(whisper_chunks[i * num_frames : (i + 1) * num_frames])
                audio_embeds = audio_embeds.to(device, dtype=weight_dtype)
                if do_classifier_free_guidance:
                    null_audio_embeds = torch.zeros_like(audio_embeds)
                    audio_embeds = torch.cat([null_audio_embeds, audio_embeds])
            else:
                audio_embeds = None
            inference_times['audio_processing'].append(time.time() - audio_start)

            # 2. Prepare faces and masks
            mask_start = time.time()
            inference_faces = faces[i * num_frames : (i + 1) * num_frames]
            latents = all_latents[:, :, i * num_frames : (i + 1) * num_frames]
            pixel_values, masked_pixel_values, masks = self.image_processor.prepare_masks_and_masked_images(
                inference_faces, affine_transform=False
            )
            inference_times['prepare_masks'].append(time.time() - mask_start)

            # 3. Prepare mask latents
            mask_latents_start = time.time()
            mask_latents, masked_image_latents = self.prepare_mask_latents(
                masks, masked_pixel_values, height, width, weight_dtype, device, generator, do_classifier_free_guidance
            )
            inference_times['prepare_mask_latents'].append(time.time() - mask_latents_start)

            # 4. Prepare image latents
            img_latents_start = time.time()
            image_latents = self.prepare_image_latents(
                pixel_values, device, weight_dtype, generator, do_classifier_free_guidance
            )
            inference_times['prepare_image_latents'].append(time.time() - img_latents_start)

            # 5. Denoising loop
            denoise_start = time.time()
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for j, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                    # concat latents, mask, masked_image_latents in the channel dimension
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    latent_model_input = torch.cat(
                        [latent_model_input, mask_latents, masked_image_latents, image_latents], dim=1
                    )

                    # predict the noise residual
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=audio_embeds).sample

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_audio - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    # call the callback, if provided
                    if j == len(timesteps) - 1 or ((j + 1) > num_warmup_steps and (j + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and j % callback_steps == 0:
                            callback(j, t, latents)
            inference_times['denoising_steps'].append(time.time() - denoise_start)

            # 6. Decode latents
            decode_start = time.time()
            decoded_latents = self.decode_latents(latents)
            inference_times['decode_latents'].append(time.time() - decode_start)

            # 7. Paste back pixels
            paste_start = time.time()
            decoded_latents = self.paste_surrounding_pixels_back(
                decoded_latents, pixel_values, 1 - masks, device, weight_dtype
            )
            inference_times['paste_pixels'].append(time.time() - paste_start)
            
            synced_video_frames.append(decoded_latents)

        # Print timing statistics
        for key, times in inference_times.items():
            avg_time = sum(times) / len(times)
            total_time = sum(times)
            print(f"{key}:")
            print(f"  Average time per iteration: {avg_time:.3f}s")
            print(f"  Total time: {total_time:.3f}s")
        
        step_5 = time.time()
        synced_video_frames = self.restore_video(
            torch.cat(synced_video_frames), original_video_frames, boxes, affine_matrices
        )
        # masked_video_frames = self.restore_video(
        #     torch.cat(masked_video_frames), original_video_frames, boxes, affine_matrices
        # )
        step_6 = time.time()

        audio_samples_remain_length = int(synced_video_frames.shape[0] / video_fps * audio_sample_rate)
        audio_samples = audio_samples[:audio_samples_remain_length].cpu().numpy()

        if is_train:
            self.unet.train()

        temp_dir = "temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        step_7 = time.time()
        write_video(os.path.join(temp_dir, "video.mp4"), synced_video_frames, fps=25)
        # write_video(video_mask_path, masked_video_frames, fps=25)

        sf.write(os.path.join(temp_dir, "audio.wav"), audio_samples, audio_sample_rate)

        command = f"ffmpeg -y -loglevel error -nostdin -i {os.path.join(temp_dir, 'video.mp4')} -i {os.path.join(temp_dir, 'audio.wav')} -c:v libx264 -c:a aac -q:v 0 -q:a 0 {video_out_path}"
        subprocess.run(command, shell=True)
        step_8 = time.time()
        print(f"step 0: {step_0-start}")
        print(f"step 1: {step_1-step_0}")
        print(f"step 2: {step_2-step_1}")
        print(f"step 3: {step_3-step_2}")
        print(f"step 4: {step_4-step_3}")
        print(f"step 5: {step_5-step_4}")
        print(f"step 6: {step_6-step_5}")
        print(f"step 7: {step_7-step_6}")
        print(f"step 8: {step_8-step_7}")
        print(f"Total time: {step_8 - start}")