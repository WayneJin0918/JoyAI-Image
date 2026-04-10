"""
JoyAI Image Edit Custom Nodes for ComfyUI
Provides nodes to load and run JoyAI Image Edit models within ComfyUI
"""

import os
import sys
import torch
import numpy as np
import folder_paths
from pathlib import Path
from PIL import Image
import comfy.model_management

# Add JoyAI-Image source to path
try:
    # JoyAI-Image is located in the same directory as custom_nodes
    joyai_root = Path(__file__).parent.parent / "JoyAI-Image"
    joyai_src = joyai_root / "src"
    if str(joyai_src) not in sys.path:
        sys.path.insert(0, str(joyai_src))
except Exception as e:
    print(f"Warning: Could not add JoyAI-Image to path: {e}")


class JoyAIImageEditTransformerLoader:
    """Load JoyAI Image Edit Transformer (MMDiT) model"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "transformer_ckpt_path": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "precision": (["fp32", "fp16", "bf16"], {"default": "fp16"}),
            }
        }

    RETURN_TYPES = ("JOYAI_TRANSFORMER",)
    RETURN_NAMES = ("transformer",)
    FUNCTION = "load_transformer"
    CATEGORY = "loaders/joyai"

    def load_transformer(self, transformer_ckpt_path, precision):
        """Load the transformer model"""
        from modules.models.mmdit.dit import Transformer3DModel
        from modules.utils.constants import PRECISION_TO_TYPE

        device = comfy.model_management.get_torch_device()
        dtype = PRECISION_TO_TYPE.get(precision, torch.float16)

        if not os.path.exists(transformer_ckpt_path):
            raise FileNotFoundError(f"Transformer checkpoint not found: {transformer_ckpt_path}")

        # Load model config - JoyAI-Image MMDiT model configuration
        # These are the correct parameters for the Transformer3DModel
        args = None
        config = {
            "hidden_size": 4096,
            "in_channels": 16,
            "heads_num": 32,
            "mm_double_blocks_depth": 40,
            "out_channels": 16,
            "patch_size": [1, 2, 2],
            "rope_dim_list": [16, 56, 56],
            "text_states_dim": 4096,
            "rope_type": "rope",
            "dit_modulation_type": "wanx",
            "theta": 10000,
            "attn_backend": "flash_attn",
        }
        model_kwargs = {'dtype': dtype, 'device': device}
        config.update(model_kwargs)

        model = Transformer3DModel(args, **config)

        # Load checkpoint
        state_dict = torch.load(transformer_ckpt_path, map_location='cpu')
        if "model" in state_dict:
            state_dict = state_dict["model"]
        if state_dict is not None:
            # filter unused params
            load_state_dict = {}
            for k, v in state_dict.items():

                if k == "img_in.weight" and model.img_in.weight.shape != v.shape:
                    print(f"Inflate {k} from {v.shape} to {model.img_in.weight.shape}")
                    v_new = v.new_zeros(model.img_in.weight.shape)
                    v_new[:, :v.shape[1], :, :, :] = v
                    v = v_new

                load_state_dict[k] = v
            model.load_state_dict(load_state_dict, strict=True)

        model = model.to(device=device, dtype=dtype)
        model.requires_grad_(False)
        model.eval()

        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Instantiate model with {total_params / 1e9:.2f}B parameters")

        # hack
        model.__dict__["precision"] = precision

        return (model,)


class JoyAIImageEditTextEncoderLoader:
    """Load JoyAI Text Encoder (Qwen3VL)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_encoder_path": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "precision": (["fp32", "fp16", "bf16"], {"default": "fp16"}),
            }
        }

    RETURN_TYPES = ("JOYAI_TEXT_ENCODER", "JOYAI_TOKENIZER")
    RETURN_NAMES = ("text_encoder", "tokenizer")
    FUNCTION = "load_text_encoder"
    CATEGORY = "loaders/joyai"

    def load_text_encoder(self, text_encoder_path, precision):
        """Load text encoder and tokenizer"""
        from modules.models.mmdit.text_encoder import load_text_encoder
        from modules.utils.constants import PRECISION_TO_TYPE

        device = comfy.model_management.get_torch_device()
        dtype = PRECISION_TO_TYPE.get(precision, torch.float16)

        if not os.path.exists(text_encoder_path):
            raise FileNotFoundError(f"Text encoder path not found: {text_encoder_path}")

        # Load text encoder and tokenizer
        tokenizer, text_encoder = load_text_encoder(text_encoder_path, device=device, torch_dtype=dtype)
        text_encoder.requires_grad_(False)
        text_encoder.eval()

        # hack
        text_encoder.__dict__["precision"] = precision
        text_encoder.__dict__["text_encoder_arch_config"] = {
            "target": "modules.models.load_text_encoder",
            "params": {
                "text_encoder_ckpt": os.path.abspath(text_encoder_path),
            },
        }

        return (text_encoder, tokenizer)


class JoyAIImageEditVAELoader:
    """Load JoyAI WanVAE for image encoding/decoding"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_ckpt_path": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "precision": (["fp32", "fp16", "bf16"], {"default": "fp16"}),
            }
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load_vae"
    CATEGORY = "loaders/joyai"

    def load_vae(self, vae_ckpt_path, precision):
        """Load the WanVAE model"""
        from modules.models.mmdit.vae import WanxVAE
        from modules.utils.constants import PRECISION_TO_TYPE

        device = comfy.model_management.get_torch_device()
        dtype = PRECISION_TO_TYPE.get(precision, torch.float16)

        if not os.path.exists(vae_ckpt_path):
            raise FileNotFoundError(f"VAE checkpoint not found: {vae_ckpt_path}")

        # Create and load WanxVAE
        # Note: WanxVAE expects pretrained path to the checkpoint
        vae = WanxVAE(
            pretrained=vae_ckpt_path,
            torch_dtype=dtype,
            device=str(device)
        )
        vae.requires_grad_(False)
        vae.eval()

        # hack
        vae.__dict__["precision"] = precision

        return (vae,)


class JoyAIImageEditPipeline:
    """JoyAI Image Edit Pipeline - performs text-to-image or image editing"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "transformer": ("JOYAI_TRANSFORMER",),
                "text_encoder": ("JOYAI_TEXT_ENCODER",),
                "tokenizer": ("JOYAI_TOKENIZER",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 1000}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2147483647, "control_after_generate": "randomize"}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 32}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 32}),
            },
            "optional": {
                "image": ("IMAGE",),
                "basesize": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 128}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "pipeline"
    CATEGORY = "image/joyai"

    def pipeline(self, transformer, text_encoder, tokenizer, vae, prompt,
                negative_prompt, steps, guidance_scale, seed, height, width,
                image=None, basesize=1024):
        """Run the JoyAI image edit pipeline"""
        from modules.models.pipeline import Pipeline
        from modules.models.scheduler import FlowMatchDiscreteScheduler
        from modules.utils import _dynamic_resize_from_bucket, seed_everything
        from types import SimpleNamespace

        device = comfy.model_management.get_torch_device()
        seed_everything(seed)

        # Create scheduler
        scheduler_arch_config = {
            "num_train_timesteps": 1000,
            "shift": 4.0,
        }
        scheduler = FlowMatchDiscreteScheduler(**scheduler_arch_config)

        # Create pipeline
        cfg = {
            "text_token_max_length": 2048,
            "dit_precision": transformer.precision,
            "vae_precision": vae.precision,
            "text_encoder_precision": text_encoder.precision,
            "text_encoder_arch_config": text_encoder.text_encoder_arch_config
        }
        cfg = SimpleNamespace(**cfg)
        pipeline = Pipeline(
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
            args=cfg,
        )

        # Prepare inputs
        if image is None:
            # Text-to-image mode
            prompts = [prompt]
            negative_prompts = [negative_prompt]
            images = None
            h, w = height, width
        else:
            # Image editing mode
            # Convert tensor to PIL
            img_array = (image[0].cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_array)

            # Resize using bucket
            processed = _dynamic_resize_from_bucket(pil_image, basesize=basesize)
            w, h = processed.size

            image_tokens = '<image>\n'
            prompts = [f"<|im_start|>user\n{image_tokens}{prompt}<|im_end|>\n"]
            negative_prompts = [f"<|im_start|>user\n{image_tokens}{negative_prompt}<|im_end|>\n"]
            images = [processed]

        # Run inference
        generator = torch.Generator(device=device).manual_seed(int(seed))

        with torch.no_grad():
            output = pipeline(
                prompt=prompts,
                negative_prompt=negative_prompts,
                images=images,
                height=h,
                width=w,
                num_frames=1,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                num_videos_per_prompt=1,
                output_type='pt',
                return_dict=False,
            )

        # Process output
        image_tensor = output[0, -1, 0]  # (C, H, W)
        image_tensor = torch.clamp(image_tensor, 0, 1)
        image_tensor = image_tensor.permute(1, 2, 0).unsqueeze(0)

        return (image_tensor,)
