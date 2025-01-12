##################################################
# Import Packages
##################################################
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2TokenizerFast
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import requests
from .utils import split_image  # Import the utility function for splitting

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

##################################################
# Load the Pretrained Model, Processor, and Tokenizer
##################################################
model_raw = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

##################################################
# Function to Generate Caption for Full Image
##################################################
def generate_caption_full_image(image, greedy=True):
    """
    Generates a caption for the full image using the traditional approach.
    """
    pixel_values = image_processor(image, return_tensors="pt").pixel_values

    if greedy:
        generated_ids = model_raw.generate(pixel_values=pixel_values, max_new_tokens=30)
    else:
        generated_ids = model_raw.generate(pixel_values=pixel_values, do_sample=True, max_new_tokens=30, top_k=5)

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text

##################################################
# Function to Generate Captions for Quadrants
##################################################
def generate_quadrant_captions(image, greedy=True):
    """
    Generates captions for each quadrant of the image using the 4CLIP approach.
    """
    quadrants = split_image(image)
    captions = []

    for quadrant in quadrants:
        caption = generate_caption_full_image(quadrant, greedy)
        captions.append(caption)

    return captions

##################################################
# Function to Generate Final Caption Using Combined Features
##################################################
def generate_final_caption(image, greedy=True):
    """
    Generates a final caption for the full image using combined features from all quadrants.
    """
    quadrants = split_image(image)
    features = []

    for quadrant in quadrants:
        pixel_values = image_processor(quadrant, return_tensors="pt").pixel_values
        features.append(pixel_values)

    combined_features = torch.cat(features, dim=0)

    if greedy:
        generated_ids = model_raw.generate(pixel_values=combined_features, max_new_tokens=30)
    else:
        generated_ids = model_raw.generate(pixel_values=combined_features, do_sample=True, max_new_tokens=30, top_k=5)

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text

##################################################
# Function to Compare Traditional and 4CLIP Captions
##################################################
def compare_captions(url, greedy=True):
    """
    Compares the traditional and 4CLIP approaches for image captioning.
    """
    image = Image.open(requests.get(url, stream=True).raw)

    # Generate traditional caption
    print("\n### Traditional Caption (Direct Image Input) ###")
    traditional_caption = generate_caption_full_image(image, greedy)
    print(f"Caption: {traditional_caption}")

    # Generate and display quadrant captions
    print("\n### 4CLIP Quadrant Captions ###")
    quadrants = split_image(image)
    quadrant_captions = generate_quadrant_captions(image, greedy)

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.flatten()

    for i, (quadrant, caption) in enumerate(zip(quadrants, quadrant_captions)):
        axs[i].imshow(np.asarray(quadrant))
        axs[i].set_title(f"Quadrant {i + 1} Caption:\n{caption}", fontsize=10)
        axs[i].axis("off")

    plt.tight_layout()
    plt.show()

    # Generate and display the final caption
    print("\n### 4CLIP Final Caption (Combined Quadrant Features) ###")
    final_caption = generate_final_caption(image, greedy)
    print(f"Final Caption: {final_caption}")

    plt.imshow(np.asarray(image))
    plt.title(f"Final Caption:\n{final_caption}", fontsize=12)
    plt.axis("off")
    plt.show()
