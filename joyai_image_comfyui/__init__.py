from .joyai_image_edit_nodes import (
    JoyAIImageEditTransformerLoader,
    JoyAIImageEditTextEncoderLoader,
    JoyAIImageEditVAELoader,
    JoyAIImageEditPipeline,
)

__all__ = [
    "JoyAIImageEditTransformerLoader",
    "JoyAIImageEditTextEncoderLoader",
    "JoyAIImageEditVAELoader",
    "JoyAIImageEditPipeline",
]

NODE_CLASS_MAPPINGS = {
    "JoyAIImageEditTransformerLoader": JoyAIImageEditTransformerLoader,
    "JoyAIImageEditTextEncoderLoader": JoyAIImageEditTextEncoderLoader,
    "JoyAIImageEditVAELoader": JoyAIImageEditVAELoader,
    "JoyAIImageEditPipeline": JoyAIImageEditPipeline,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JoyAIImageEditTransformerLoader": "Load JoyAI Image Edit Transformer",
    "JoyAIImageEditTextEncoderLoader": "Load JoyAI Text Encoder",
    "JoyAIImageEditVAELoader": "Load JoyAI WanVAE",
    "JoyAIImageEditPipeline": "JoyAI Image Edit Pipeline",
}
