from transformers.tools.base import Tool, get_default_device
from transformers.utils import is_accelerate_available
import torch

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

class StableDiffusion():
    default_checkpoint = "Ojimi/anime-kawai-diffusion"

    def __init__(self, device=None, **hub_kwargs) -> None:
        if not is_accelerate_available():
            raise ImportError(
                "Accelerate should be installed in order to use tools.")

        super().__init__()

        self.device = device
        self.pipeline = None
        self.hub_kwargs = hub_kwargs
        self.is_initialized = False

    def setup(self):
        if self.device is None:
            self.device = get_default_device()

        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.default_checkpoint, device=self.device, **self.hub_kwargs, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipeline.scheduler.config)
        self.pipeline.to(self.device)

        if self.device.type == "cuda":
            self.pipeline.to(torch_dtype=torch.float16)

        self.is_initialized = True

    def __call__(self, prompt):
        if not self.is_initialized:
            self.setup()

        negative_prompt = "(worst quality, low quality:1.4), ((bad anatomy, short legs)), bad hands, cropped, missing fingers, ((missing toes, too many toes, extra toes, extra digits)), too many fingers, missing arms, long neck, ((missing legs, too many legs, extra legs, missing feet, extra feet, too many feet)), Humpbacked, (deformed, disfigured), poorly drawn face, distorted face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb, floating limbs, disconnected limbs, malformed hands, out of focus, long body, monochrome, symbol, text, logo, door frame, window frame, mirror frame, text box, speech balloons, blurry, censored, (bar censor:1.2), mosaic censoring, ((signature:1.4)), (((text:1.2))), (((many girls:1.6))), ((sadness expressions:1.6)), ((sad expressions:1.6)), ((kid:1.2)), ((loli:1.2)), ((children:1.2))"
        added_prompt = " , highest quality, highly realistic, very high resolution, masterpiece, best quality"

        return self.pipeline(prompt + added_prompt, negative_prompt=negative_prompt, num_inference_steps=55).images[0]