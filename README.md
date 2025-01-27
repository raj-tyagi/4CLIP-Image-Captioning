# 4CLIP: Enhanced Image Captioning using Quadrant-Based Feature Extraction

4CLIP is a research-driven image captioning project that enhances traditional image captioning models by dividing images into four quadrants and processing them individually. This quadrant-based approach allows for better image understanding and more comprehensive captions. The project utilizes a pretrained VisionEncoderDecoderModel from the Hugging Face Transformers library, specifically the ViT-GPT2 model. 

## Repository Name
`4CLIP-Image-Captioning`

## Project Structure
```
├── README.md                      # Detailed project description (this file)
├── requirements.txt               # List of necessary packages for the project
├── src/                           # Source code directory containing Python scripts
│   ├── image_captioning.py        # Core script containing image captioning functions
│   └── utils.py                   # Utility functions for image processing
└── LICENSE                        # Project license
```

## Installation
To run this project, ensure you have Python installed. Then, install the necessary packages:
```bash
pip install transformers torch requests pillow matplotlib tqdm
```

## Usage
You can run the project directly through the Jupyter Notebook provided (`4clip_image_captioning.ipynb`).

```python
# Example Usage:
from src.image_captioning import compare_captions
compare_captions(url="https://example.com/sample.jpg", greedy=True)
```

### Functionality
1. **Full Image Captioning:** Captions generated for the entire image.
2. **Quadrant Captioning:** The image is split into four quadrants, and captions are generated for each.
3. **Final Caption (4CLIP):** Captions generated using combined quadrant features.

## How It Works
### Steps Implemented:
1. **Image Splitting:** The image is divided into four quadrants.
2. **Feature Extraction:** Features are extracted using the ViT processor.
3. **Caption Generation:**
   - Traditional Caption: Generated using the full image.
   - Quadrant Captions: Each quadrant is captioned separately.
   - Final Caption: Quadrant features are combined and passed into the model for a comprehensive caption.

## Example Results
**Traditional Caption:**
> "A flag waving in the wind."

**4CLIP Captions:**
- Quadrant 1: "A red and yellow flag."
- Quadrant 2: "Blue sky in the background."
- Quadrant 3: "A waving flag pole."
- Quadrant 4: "A bright outdoor scene."

**Final Caption (4CLIP):**
> "A red and yellow flag waving in the wind with a bright blue sky."

## Research Objective
The aim of this project is to explore whether splitting an image into multiple parts and combining their features can generate more detailed and descriptive captions. This could be particularly useful in domains requiring fine-grained visual understanding, such as medical imaging and assistive technologies.

## Contributing
Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments
- Hugging Face for the pretrained ViT-GPT2 model.
- Python libraries: PyTorch, Transformers, Pillow, Matplotlib.

---

**Raj Tyagi**: [LinkedIn](https://www.linkedin.com/in/raj-tyagi-83765b21b/) 

