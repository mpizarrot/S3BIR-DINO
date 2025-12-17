import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from s3bir_dinov3_model import S3birDinov3

def make_transform():
    to_tensor = v2.ToImage()
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, to_float, normalize])

def visualize_cls_patch(img, sketch, cls_img, cls_sketch, patch_img, patch_sketch):
    # CLS-to-patches y patches-to-CLS
    sim_cls_img_patch_sketch = F.cosine_similarity(cls_img.unsqueeze(1).expand(-1, patch_sketch.shape[1], -1), patch_sketch, dim=-1)
    sim_cls_sketch_patch_img = F.cosine_similarity(cls_sketch.unsqueeze(1).expand(-1, patch_img.shape[1], -1), patch_img, dim=-1)

    map_size = int(np.sqrt(patch_img.shape[1]))
    map12 = sim_cls_img_patch_sketch.view(map_size, map_size).detach().cpu().numpy()
    map21 = sim_cls_sketch_patch_img.view(map_size, map_size).detach().cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(8, 7))

    axes[0, 0].imshow(img)
    axes[0, 0].set_title("image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(sketch)
    axes[0, 1].set_title("sketch")
    axes[0, 1].axis("off")

    im2 = axes[1, 0].imshow(map21, cmap="viridis")
    axes[1, 0].set_title("Similarity: CLS(sketch) -> patches(image)")
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im1 = axes[1, 1].imshow(map12, cmap="viridis")
    axes[1, 1].set_title("Similarity: CLS(image) -> patches(sketch)")
    plt.colorbar(im1, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

def infer_grid(n_tokens: int, h_hint: int = None, w_hint: int = None):
    """Infiera grid HxW; si no hay hints, asume cuadrado."""
    if h_hint is not None and w_hint is not None:
        assert h_hint * w_hint == n_tokens, "Hints de grid no coinciden con N tokens."
        return h_hint, w_hint
    h = int(np.sqrt(n_tokens))
    if h * h == n_tokens:
        return h, h
    # fallback: busca factores
    for hh in range(h, 1, -1):
        if n_tokens % hh == 0:
            return hh, n_tokens // hh
    raise ValueError("No se pudo inferir un grid HxW válido para N tokens.")

def idx_to_rc(idx: int, grid_h: int, grid_w: int):
    """Convierte índice lineal a (row, col)."""
    assert 0 <= idx < grid_h * grid_w, "Índice fuera de rango."
    r = idx // grid_w
    c = idx % grid_w
    return r, c

def plot_with_red_dot(ax, pil_img, idx: int, grid_h: int, grid_w: int, title: str):
    """Muestra imagen y dibuja punto en el centro del patch idx."""
    ax.imshow(pil_img)
    ax.set_title(title)
    ax.axis("off")
    w, h = pil_img.size
    cell_w = w / grid_w
    cell_h = h / grid_h
    r, c = idx_to_rc(idx, grid_h, grid_w)
    x = (c + 0.5) * cell_w
    y = (r + 0.5) * cell_h
    ax.scatter([x], [y], s=60, c="red", marker="o", edgecolors="white", linewidths=1.5)

def patch_to_patches_similarity(query_patch_vec: torch.Tensor,
                                target_patches: torch.Tensor) -> torch.Tensor:
    """
    query_patch_vec: (D,) o (1,D)
    target_patches: (1, N, D)
    retorna: (N,) similitudes coseno
    """
    if query_patch_vec.dim() == 1:
        query_patch_vec = query_patch_vec.unsqueeze(0)  # (1, D)
    query_expand = query_patch_vec.unsqueeze(1).expand(-1, target_patches.shape[1], -1)  # (1, N, D)
    sim = F.cosine_similarity(target_patches, query_expand, dim=-1)  # (1, N)
    return sim.squeeze(0)  # (N,)

def visualize_patch_matching(img, sketch,
                             patch_img, patch_sketch,
                             img_patch_idx: int,
                             sketch_patch_idx: int,
                             grid_img: tuple = None,
                             grid_sketch: tuple = None,
                             cmap="viridis"):
    """
      [0,0] Imagen con punto (img_patch_idx)
      [0,1] Sketch con punto (sketch_patch_idx)
      [1,0] fmap: patch(Imagen) → patches(Sketch)
      [1,1] fmap: patch(Sketch) → patches(Imagen)
    """
    # Grids
    N_img = patch_img.shape[1]
    N_ske = patch_sketch.shape[1]
    H_img, W_img = infer_grid(N_img, *(grid_img or (None, None)))
    H_ske, W_ske = infer_grid(N_ske, *(grid_sketch or (None, None)))

    # Similitud: patch(Imagen) -> patches(Sketch)
    q_vec_img = patch_img[0, img_patch_idx, :]  # (D,)
    sim_img2ske = patch_to_patches_similarity(q_vec_img, patch_sketch)  # (N_ske,)
    fmap_img2ske = sim_img2ske.view(H_ske, W_ske).detach().cpu().numpy()

    # Similitud: patch(Sketch) -> patches(Imagen)
    q_vec_ske = patch_sketch[0, sketch_patch_idx, :]  # (D,)
    sim_ske2img = patch_to_patches_similarity(q_vec_ske, patch_img)  # (N_img,)
    fmap_ske2img = sim_ske2img.view(H_img, W_img).detach().cpu().numpy()

    # Figura
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))

    plot_with_red_dot(axes[0, 0], img, img_patch_idx, H_img, W_img, "image (selected patch)")
    plot_with_red_dot(axes[0, 1], sketch, sketch_patch_idx, H_ske, W_ske, "sketch (selected patch)")

    im_right = axes[1, 0].imshow(fmap_ske2img, cmap=cmap)
    axes[1, 0].set_title("Similarity: patch(sketch) -> patches(image)")
    plt.colorbar(im_right, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    im_left = axes[1, 1].imshow(fmap_img2ske, cmap=cmap)
    axes[1, 1].set_title("Similarity: patch(image) -> patches(sketch)")
    plt.colorbar(im_left, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ckpt_path = "/home/shared_data/s3bir/saved_models/skDinoV3_sketchy.ckpt"
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = S3birDinov3().to(device)
    model.load_state_dict(state_dict)
    model.eval()

    img = Image.open("/home/data/Datasets/sketchy/photo/airplane/n02691156_10578.jpg").convert("RGB")
    sketch = Image.open("/home/data/Datasets/sketchy/sketch/airplane/n02691156_10151-1.png").convert("RGB")

    transform = make_transform()
    img_tensor = transform(img).unsqueeze(0).to(device)     # [1, 3, 224, 224]
    sketch_tensor = transform(sketch).unsqueeze(0).to(device)

    with torch.no_grad():
        img_embedding = model(img_tensor, dtype="image", is_training=True)
        sketch_embedding = model(sketch_tensor, dtype="sketch", is_training=True)
        
    cls_img = img_embedding['x_norm_clstoken']
    cls_sketch = sketch_embedding['x_norm_clstoken']
    patch_img = img_embedding['x_norm_patchtokens']
    patch_sketch = sketch_embedding['x_norm_patchtokens']

    cls_np = cls_img.squeeze().cpu().numpy()
    patch_np = patch_sketch.squeeze().cpu().numpy()

    visualize_cls_patch(img, sketch, 
                        cls_img, cls_sketch, 
                        patch_img, patch_sketch)
    
    visualize_patch_matching(
    img, sketch,
    patch_img, patch_sketch,
    img_patch_idx=100, #99 120
    sketch_patch_idx=148 #60 163 184
)