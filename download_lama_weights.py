#!/usr/bin/env python3
"""
Download LaMa weights from HuggingFace Hub in safetensors format.
"""

import os
import sys
from pathlib import Path
import torch

try:
    from safetensors.torch import load_file, save_file
    from huggingface_hub import hf_hub_download, HfApi
except ImportError as e:
    print(f"Error: Missing required packages. Please install them with:")
    print(f"  pip install safetensors huggingface_hub")
    sys.exit(1)


def search_repositories(query="big-lama"):
    """Search HuggingFace Hub for repositories matching the query."""
    try:
        api = HfApi()
        results = api.list_models(search=query)
        print(f"\nFound {len(results)} repositories matching '{query}':")
        for i, model in enumerate(results[:10], 1):  # Show first 10
            print(f"  {i}. {model.id}")
        return results
    except Exception as e:
        print(f"Could not search repositories: {e}")
        return []


def download_lama_weights(output_dir=None, repo_id=None, filename=None):
    """
    Download LaMa weights from HuggingFace Hub.
    
    Args:
        output_dir: Directory to save the weights. If None, saves to 'pretrained_models/big-lama'
        repo_id: HuggingFace repository ID. If None, tries common repositories.
        filename: Filename in the repository. If None, tries common filenames.
    """
    if output_dir is None:
        output_dir = Path("pretrained_models/big-lama")
    else:
        output_dir = Path(output_dir)
    
    # Create output directory structure
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Try different repositories if not specified
    if repo_id is None:
        # Try common repository names
        repo_candidates = [
            "michaelgold/big-lama-safetensors",
            "big-lama-safetensors",
            "big-lama",
            "lama-safetensors",
        ]
    else:
        repo_candidates = [repo_id]
    
    # Try different filenames if not specified
    if filename is None:
        filename_candidates = [
            "big-lama.safetensors",
            "best.ckpt",
            "big-lama.ckpt",
            "model.safetensors",
        ]
    else:
        filename_candidates = [filename]
    
    print(f"Downloading LaMa weights from HuggingFace Hub...")
    print(f"Output directory: {output_dir}")
    
    # Try each repository and filename combination
    for repo_id in repo_candidates:
        for filename in filename_candidates:
            try:
                print(f"\nTrying repository: {repo_id}, filename: {filename}")
                weights_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=str(output_dir),
                )
                
                print(f"✓ Downloaded weights to: {weights_path}")
                
                # Check if we need to convert or move the file
                downloaded_file = Path(weights_path)
                target_file = models_dir / "best.ckpt"
                
                # Load safetensors and save as regular checkpoint if needed
                # (LaMa expects .ckpt files, but we can also work with safetensors)
                if downloaded_file.suffix == ".safetensors":
                    print(f"Loading safetensors file...")
                    state_dict = load_file(str(downloaded_file))
                    
                    # Save as regular PyTorch checkpoint (LaMa's load_checkpoint expects this)
                    print(f"Converting to PyTorch checkpoint format...")
                    torch.save(state_dict, target_file)
                    print(f"✓ Saved checkpoint to: {target_file}")
                    
                    # Also keep the safetensors version for reference
                    safetensors_target = models_dir / "big-lama.safetensors"
                    if downloaded_file != safetensors_target:
                        import shutil
                        shutil.copy2(downloaded_file, safetensors_target)
                        print(f"✓ Copied safetensors to: {safetensors_target}")
                else:
                    # If it's already a .ckpt file, just move it
                    import shutil
                    if downloaded_file != target_file:
                        shutil.move(str(downloaded_file), str(target_file))
                    print(f"✓ Checkpoint saved to: {target_file}")
                
                # Also copy config.yaml if it exists in the repo
                try:
                    config_path = hf_hub_download(
                        repo_id=repo_id,
                        filename="config.yaml",
                        local_dir=str(output_dir),
                    )
                    print(f"✓ Downloaded config.yaml to: {config_path}")
                except:
                    pass  # Config might not exist, that's okay
                
                print(f"\n✓ Successfully downloaded and set up LaMa weights!")
                print(f"\nYou can now use the model with:")
                print(f"  --lama_ckpt {output_dir}")
                
                return str(output_dir)
                
            except Exception as e:
                # Try next combination
                continue
    
    # If we get here, all attempts failed
    print(f"\n✗ Error: Could not download weights from any repository.")
    print(f"\nTried repositories: {repo_candidates}")
    print(f"Tried filenames: {filename_candidates}")
    print(f"\nPlease specify a valid repository:")
    print(f"  python download_lama_weights.py --repo_id YOUR_REPO_ID --filename YOUR_FILENAME")
    print(f"\nOr search for repositories:")
    print(f"  python download_lama_weights.py --search")
    
    # Offer to search
    print(f"\nWould you like to search for 'big-lama' repositories? (This requires internet access)")
    search_repositories("big-lama")
    
    sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download LaMa weights from HuggingFace Hub"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for weights (default: pretrained_models/big-lama)"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="HuggingFace repository ID (e.g., 'username/repo-name')"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Filename in the repository (e.g., 'big-lama.safetensors')"
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="Search for 'big-lama' repositories on HuggingFace Hub"
    )
    
    args = parser.parse_args()
    
    if args.search:
        search_repositories("big-lama")
    else:
        download_lama_weights(args.output_dir, args.repo_id, args.filename)

