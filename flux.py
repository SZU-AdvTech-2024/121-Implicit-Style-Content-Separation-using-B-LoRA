import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("/mnt/d/studying-code/modelscope/FLUX.1-dev", torch_dtype=torch.bfloat16)
# pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

# casting here instead of in the pipeline constructor because doing so in the constructor loads all models into CPU memory at once
pipe.to(torch.float16)

# prompt = ["A photo of a bunny", "A photo of a tiger"]
prompt = ["A photo of a bunny"]

image = pipe(
    prompt,
    guidance_scale=0.0,
    num_inference_steps=4,
    max_sequence_length=256,
    height=1024,
    width=1024,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save("image.png")