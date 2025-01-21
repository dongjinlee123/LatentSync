#!/bin/bash

python -m scripts.inference \
    --unet_config_path "configs/unet/second_stage.yaml" \
    --inference_ckpt_path "checkpoints/latentsync_unet.pt" \
    --inference_steps 3 \
    --guidance_scale 1.5 \
    --video_path "assets/inference.mp4" \
    --audio_path "assets/q1.wav" \
    --video_out_path "video_out.mp4"
