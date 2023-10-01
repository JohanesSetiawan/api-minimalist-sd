from fastapi import FastAPI
from pydantic import BaseModel

from StableDiffusion import StableDiffusion
from PIL import Image
import os
import base64

class generatedImage(BaseModel):
    prompt: str

# Counter untuk nama file gambar
image_counter = 1

app = FastAPI()

@app.get("/welcome")
def welcome_api():
    return {"message": "Welcome to the FastAPI"}

@app.post("/api/img")
async def create_generated_image(item: generatedImage):
    global image_counter

    stable_diffusion = StableDiffusion()
    generated_image = stable_diffusion(item.prompt)

    img_path = os.path.join("results", f"img{image_counter}.png")
    generated_image.save(img_path)

    image_counter += 1

    # convert image to base64 encoding
    with open(img_path, "rb") as img_file:
        image_binary = img_file.read()
    image_base64 = base64.b64encode(image_binary).decode()

    # response
    return {
        "message": "Image generated and sent successfully",
        "image": image_base64
    }