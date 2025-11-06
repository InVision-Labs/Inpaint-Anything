import torch
import sys
import argparse
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

from lama_inpaint import inpaint_img_with_lama
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask


def setup_args(parser):
    parser.add_argument(
        "--input_img", type=str, required=True,
        help="Path to a single input img",
    )
    # parser.add_argument(
    #     "--coords_type", type=str, required=True,
    #     default="key_in", choices=["click", "key_in"], 
    #     help="The way to select coords",
    # )
    # parser.add_argument(
    #     "--point_coords", type=float, nargs='+', required=True,
    #     help="The coordinate of the point prompt, [coord_W coord_H].",
    # )
    # parser.add_argument(
    #     "--point_labels", type=int, nargs='+', required=True,
    #     help="The labels of the point prompt, 1 or 0.",
    # )
    parser.add_argument(
        "--dilate_kernel_size", type=int, default=None,
        help="Dilate kernel size. Default: None",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output path to the directory with results.",
    )
    # parser.add_argument(
    #     "--sam_model_type", type=str,
    #     default="vit_h", choices=['vit_h', 'vit_l', 'vit_b', 'vit_t'],
    #     help="The type of sam model to load. Default: 'vit_h"
    # )
    # parser.add_argument(
    #     "--sam_ckpt", type=str, required=True,
    #     help="The path to the SAM checkpoint to use for mask generation.",
    # )
    parser.add_argument(
        "--lama_config", type=str,
        default="./lama/configs/prediction/default.yaml",
        help="The path to the config file of lama model. "
             "Default: the config of big-lama",
    )
    parser.add_argument(
        "--lama_ckpt", type=str, required=True,
        help="The path to the lama checkpoint.",
    )
    parser.add_argument(
        "--mask", type=str, required=True,
        help="Path to a single mask image file.",
    )


if __name__ == "__main__":
    """Example usage:
    # Using a single mask file:
    python remove_anything.py \
        --input_img FA_demo/FA1_dog.png \
        --mask ./mask.png \
        --dilate_kernel_size 15 \
        --output_dir ./results \
        --lama_config lama/configs/prediction/default.yaml \
        --lama_ckpt big-lama 
    """
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load image
    img = load_img_to_array(args.input_img)

    # Load mask
    assert Path(args.mask).exists(), f"Mask file does not exist: {args.mask}"
    mask = load_img_to_array(args.mask)
    
    # Convert to grayscale if needed
    if len(mask.shape) == 3:
        mask = mask[:, :, 0] if mask.shape[2] == 1 else np.mean(mask, axis=2)
    
    # Normalize to 0-255 range if needed
    if np.max(mask) <= 1.0:
        mask = (mask * 255).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)

    # Dilate mask to avoid unmasked edge effect
    if args.dilate_kernel_size is not None:
        mask = dilate_mask(mask, args.dilate_kernel_size)

    # Create output directory
    img_stem = Path(args.input_img).stem
    out_dir = Path(args.output_dir) / img_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the mask
    mask_p = out_dir / "mask.png"
    save_array_to_img(mask, mask_p)

    # Save the masked image visualization
    img_mask_p = out_dir / "with_mask.png"
    dpi = plt.rcParams['figure.dpi']
    height, width = img.shape[:2]
    plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
    plt.imshow(img)
    plt.axis('off')
    show_mask(plt.gca(), mask, random_color=False)
    plt.savefig(img_mask_p, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Inpaint the masked image
    img_inpainted_p = out_dir / "inpainted.png"
    img_inpainted = inpaint_img_with_lama(
        img, mask, args.lama_config, args.lama_ckpt, device=device)
    save_array_to_img(img_inpainted, img_inpainted_p)
    
    print(f"✓ Inpainting complete!")
    print(f"  Input image: {args.input_img}")
    print(f"  Mask: {args.mask}")
    print(f"  Output: {img_inpainted_p}")
