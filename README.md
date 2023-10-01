## StableDiffusion API (AI Generated Images)

StableDiffusion is a state-of-the-art text to image to create an image based own your prompt. Using code from CompVis diffusions algorithms You can create and resulting beautiful image and it's free. For framework API, i used FastAPI to create an API to generate images based prompt.

## Installation

1. clone this repo
2. create a virtual environment

```python
python -m venv venv
```

3. install requirements

```python
pip install accelerate diffusers transformers torch fastapi uvicorn
```

or using requirements.txt (i don't recommend this, because it's not updated)

```python
pip install -r requirements.txt
```

Note: for torch, installed using version of compute platform compatible with your GPU. For example, if you have a GPU with compute capability CUDA 11.8, install the version of PyTorch using `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` and etc. Docs installation are available [here](https://pytorch.org/get-started/locally/)

## Usage

to run this API Server, you can run this command

```bash
uvicorn main:app --reload
```

and open your browser and go to `https://localhost:8000/docs` to see the docs and try it out.

You can change the paramater like, `negative_prompt` or `num_inference_steps` to change the result of the image, on `StableDiffusion.py` file.

additional: create a folder named 'results' to save the generated images. The folder is located on the root of this project.

## Model or Checkpoint Changing

You can custom the model by searching on [HuggingFace](https://huggingface.co/models). For example, i use `Ojimi/anime-kawai-diffusion` model to generate images. You can change the model by changing the `default_checkpoints` variable in `StableDiffusion.py` file.

example:

```python
default_checkpoints = "Ojimi/anime-kawai-diffusion"
```

source model or checkpoints [here](https://huggingface.co/Ojimi/anime-kawai-diffusion)

## Contributing

[Me](https://github.com/JohanesSetiawan) and [My Friend](https://github.com/masnajeeeeb27)
