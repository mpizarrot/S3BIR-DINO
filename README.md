# Extract embeddings with `S3BIR-DINOv2`

This repository provides a simple pipeline to extract embeddings from images and sketches using our S3BIR-DINOv2 model.
(Prereqs: Python 3.9+, `torch`, `torchvision`, `Pillow`.)


## 1) Download the checkpoint
You can download the checkpoint from [this link](http://201.238.213.114:2280/sketchapp/get_files/s3bir_dinov2_flickr.ckpt).

## 2) Load the model

```python
import torch
from model import S3birDinov2

ckpt_path = "s3bir_dinov2_flickr.ckpt"
checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
state_dict = checkpoint.get("state_dict", checkpoint)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = S3birDinov2().to(device)
model.load_state_dict(state_dict)
model.eval()

print("Model loaded and set to eval mode.")
```

## 3) Load and transform images

```python
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

img = Image.open("image.jpg").convert("RGB")
sketch = Image.open("sketch.png").convert("RGB")

img_tensor = transform(img).unsqueeze(0).to(device)     # [1, 3, 224, 224]
sketch_tensor = transform(sketch).unsqueeze(0).to(device)
```

## 4) Compute embeddings

```python
import torch

with torch.no_grad():
    # The model expects a dtype flag to distinguish modalities
    img_embedding = model(img_tensor, dtype="image")         # shape: [1, D]
    sketch_embedding = model(sketch_tensor, dtype="sketch")  # shape: [1, D]

print("Image embedding shape:", tuple(img_embedding.shape))
print("Sketch embedding shape:", tuple(sketch_embedding.shape))
```

> Tip: to compare embeddings, you can use cosine similarity:

```python
import torch.nn.functional as F

cos_sim = F.cosine_similarity(img_embedding, sketch_embedding)
print("Cosine similarity:", cos_sim.item())
```
