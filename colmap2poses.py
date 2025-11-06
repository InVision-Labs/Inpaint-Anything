# Confidence: 85%
# colmap2poses.py — Generate LLFF-compatible poses_bounds.npy from COLMAP sparse/0
#
# Core behavior:
# - Reads COLMAP binary or text model from <scene_root>/sparse/0 using pycolmap
# - Emits <scene_root>/poses_bounds.npy with shape (N, 17)
#   For each registered image i:
#     - The first 15 numbers reshape to (3,5):
#         [c2w[:,0]  c2w[:,1]  c2w[:,2]  t_c2w  [H, W, f]]
#       where c2w is camera-to-world rotation (columns) and t is translation.
#     - The last 2 numbers are near_i, far_i (depth bounds from observed 3D points).
#
# Requirements:
#   pip install pycolmap numpy
#
# Usage:
#   python colmap2poses.py <scene_root>
#
# Notes:
# - Intrinsics are taken per-image from its camera; f = (fx + fy) / 2.
# - Images are ordered lexicographically by image name for reproducibility.
# - Depth bounds are computed from the z-depths of 3D points observed in each view
#   using robust percentiles [0.1, 99.9]. Adjust with --near_perc/--far_perc if needed.

import argparse
import sys
from pathlib import Path
from typing import Tuple, List

import numpy as np

try:
    import pycolmap  # type: ignore
except Exception as e:
    print("ERROR: pycolmap is required. Install with: pip install pycolmap", file=sys.stderr)
    raise

# ---------------------------- math helpers ---------------------------- #

def qvec2rotmat(q: np.ndarray) -> np.ndarray:
    """Convert COLMAP (qw, qx, qy, qz) to 3x3 rotation matrix (world->cam)."""
    assert q.shape == (4,)
    qw, qx, qy, qz = q
    # normalize
    n = np.linalg.norm(q)
    if n == 0:
        raise ValueError("Zero-norm quaternion encountered.")
    qw, qx, qy, qz = q / n
    R = np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
    ], dtype=np.float64)
    return R


def world_to_cam(Rcw: np.ndarray, tcw: np.ndarray, Xw: np.ndarray) -> np.ndarray:
    """Apply world->camera transform."""
    return Rcw @ Xw + tcw


def cam_to_world(Rcw: np.ndarray, tcw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return camera-to-world rotation and translation."""
    Rc2w = Rcw.T
    tc2w = -Rc2w @ tcw
    return Rc2w, tc2w


# ---------------------------- intrinsics ---------------------------- #

def focal_from_camera(cam: pycolmap.Camera) -> Tuple[float, float, float, float]:
    """
    Extract fx, fy, cx, cy from common COLMAP camera models.
    Falls back to symmetric focal if only one f is present.
    """
    model = cam.model
    params = cam.params
    # Map common models
    if model in ("SIMPLE_PINHOLE",):
        f, cx, cy = params
        fx = fy = float(f)
    elif model in ("PINHOLE",):
        fx, fy, cx, cy = params
    elif model in ("SIMPLE_RADIAL", "RADIAL", "OPENCV", "OPENCV_FISHEYE",
                   "FULL_OPENCV", "FOV", "THIN_PRISM_FISHEYE", "DIVISION",
                   "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
        # First element is focal or fx; many models assume fx=fy for principal focal.
        # Use fx=fy=first focal-like parameter; then pull cx, cy from their usual slots.
        # Model parameter layouts vary; use documented positions where available.
        # Safe fallback:
        f = float(params[0])
        fx = fy = f
        # COLMAP typically stores cx, cy near the end of the intrinsic block.
        if len(params) >= 3:
            cx = float(params[-2])
            cy = float(params[-1])
        else:
            # If cx,cy absent, center of image will be used downstream if needed.
            cx = float(cam.width) * 0.5
            cy = float(cam.height) * 0.5
    else:
        # Unknown model: best-effort symmetric focal and centered principal point.
        fx = fy = float(params[0]) if len(params) > 0 else 0.5 * (cam.width + cam.height)
        cx = float(cam.width) * 0.5
        cy = float(cam.height) * 0.5

    return float(fx), float(fy), float(cx), float(cy)


# ---------------------------- main logic ---------------------------- #

def build_poses_bounds(
    recon: pycolmap.Reconstruction,
    near_perc: float,
    far_perc: float,
) -> Tuple[np.ndarray, List[str]]:
    """
    Construct LLFF poses_bounds array and return image name ordering.
    Returns:
      poses_bounds: (N, 17)
      names: list of image names in the same order
    """
    if len(recon.images) == 0:
        raise ValueError("No registered images found in the reconstruction.")
    if len(recon.points3D) == 0:
        raise ValueError("No 3D points found; run triangulation before export.")

    # Sort by image name for deterministic ordering
    images_sorted = sorted(recon.images.values(), key=lambda im: im.name)
    names = [im.name for im in images_sorted]
    N = len(images_sorted)

    # Prepare outputs
    poses_3x5 = np.zeros((N, 3, 5), dtype=np.float64)
    bounds_2 = np.zeros((N, 2), dtype=np.float64)

    # For quick 3D point lookup
    points3D = recon.points3D

    for i, im in enumerate(images_sorted):
        # World->Cam
        if hasattr(im, 'qvec') and hasattr(im, 'tvec'):
            q = np.asarray(im.qvec, dtype=np.float64).reshape(4,)
            t = np.asarray(im.tvec, dtype=np.float64).reshape(3,)
            Rcw = qvec2rotmat(q)  # world->cam
        else:
            # Fallbacks for newer pycolmap APIs
            if hasattr(im, 'cam_from_world'):
                Tcw_attr = im.cam_from_world
                Tcw_obj = Tcw_attr() if callable(Tcw_attr) else Tcw_attr
                M = None
                # Prefer a direct matrix method if available (e.g., Rigid3d.matrix())
                if hasattr(Tcw_obj, 'matrix') and callable(getattr(Tcw_obj, 'matrix')):
                    M = np.asarray(Tcw_obj.matrix(), dtype=np.float64)
                else:
                    # Try to treat it as an array-like first
                    try:
                        M_arr = np.asarray(Tcw_obj, dtype=np.float64)
                        if M_arr.ndim == 2 and M_arr.shape in [(4, 4), (3, 4)]:
                            M = M_arr
                    except Exception:
                        M = None
                if M is None:
                    # Compose from rotation / translation if present
                    R_mat = None
                    t_vec = None
                    if hasattr(Tcw_obj, 'rotation'):
                        R_attr = Tcw_obj.rotation
                        R_obj = R_attr() if callable(R_attr) else R_attr
                        if hasattr(R_obj, 'matrix') and callable(getattr(R_obj, 'matrix')):
                            R_mat = np.asarray(R_obj.matrix(), dtype=np.float64)
                        else:
                            R_mat = np.asarray(R_obj, dtype=np.float64).reshape(3, 3)
                    if hasattr(Tcw_obj, 'translation'):
                        t_attr = Tcw_obj.translation
                        t_obj = t_attr() if callable(t_attr) else t_attr
                        t_vec = np.asarray(t_obj, dtype=np.float64).reshape(3,)
                    if R_mat is not None and t_vec is not None:
                        M = np.eye(4, dtype=np.float64)
                        M[:3, :3] = R_mat
                        M[:3, 3] = t_vec
                if M is None:
                    raise AttributeError("Unable to extract cam_from_world as matrix from pycolmap Image")
                if M.shape == (4, 4):
                    Rcw = M[:3, :3]
                    t = M[:3, 3]
                elif M.shape == (3, 4):
                    Rcw = M[:3, :3]
                    t = M[:3, 3]
                else:
                    raise AttributeError("Unexpected cam_from_world matrix shape; expected 3x4 or 4x4")
            elif hasattr(im, 'rotation') and hasattr(im, 'translation'):
                Rcw = np.asarray(im.rotation, dtype=np.float64).reshape(3, 3)
                t = np.asarray(im.translation, dtype=np.float64).reshape(3,)
            else:
                raise AttributeError("pycolmap Image missing pose attributes (qvec/tvec or cam_from_world/rotation/translation)")
        Rc2w, tc2w = cam_to_world(Rcw, t)

        # Intrinsics
        cam = recon.cameras[im.camera_id]
        fx, fy, cx, cy = focal_from_camera(cam)
        f = 0.5 * (fx + fy)
        H, W = int(cam.height), int(cam.width)

        # Pose block (3x5)
        # Columns: [right, up, back, translation, [H, W, f] as a column vector]
        # Note: COLMAP's camera looks along +Z in its camera coords; with Rc2w derived from Rcw,
        # the world "back" direction (third column) equals Rc2w[:, 2].
        poses_3x5[i, :, 0:3] = Rc2w  # columns are the c2w axes
        poses_3x5[i, :, 3] = tc2w
        poses_3x5[i, :, 4] = np.array([float(H), float(W), float(f)], dtype=np.float64)

        # Depth bounds from points observed in this image
        # Use only points actually observed by this image for robustness
        zs = []
        for p2d in im.points2D:
            # Handle pycolmap sentinel for "no point" which can be -1 or uint64 max
            pid_raw = p2d.point3D_id
            try:
                pid = int(pid_raw)
            except Exception:
                # Fallback: skip if cannot interpret as int
                continue
            if pid == -1 or pid == 18446744073709551615 or pid not in points3D:
                continue
            Xw = np.array(points3D[pid].xyz, dtype=np.float64)
            Xc = world_to_cam(Rcw, t, Xw)  # camera coords
            z = Xc[2]
            if np.isfinite(z):
                zs.append(z)

        if len(zs) < 2:
            # Fallback: transform all 3D points (slower)
            zs = []
            for p in points3D.values():
                Xc = world_to_cam(Rcw, t, np.asarray(p.xyz, dtype=np.float64))
                z = Xc[2]
                if np.isfinite(z):
                    zs.append(z)

        if len(zs) == 0:
            raise ValueError(f"No valid depth samples for image '{im.name}'. Cannot compute bounds.")

        zs = np.array(zs, dtype=np.float64)
        # Keep only positive depths
        zs = zs[zs > 0]
        if zs.size == 0:
            raise ValueError(f"All depths are non-positive for image '{im.name}'.")

        near = float(np.percentile(zs, near_perc))
        far = float(np.percentile(zs, far_perc))
        # Guard for tiny intervals
        eps = 1e-3
        if far - near < eps:
            mid = 0.5 * (near + far)
            near = max(eps, mid - 0.5)
            far = mid + 0.5

        bounds_2[i, 0] = near
        bounds_2[i, 1] = far

    # Stack to (N, 17)
    poses_flat = poses_3x5.reshape(N, 15)
    poses_bounds = np.concatenate([poses_flat, bounds_2], axis=1)
    return poses_bounds, names


def main():
    parser = argparse.ArgumentParser(description="Generate LLFF poses_bounds.npy from COLMAP sparse/0.")
    parser.add_argument("scene_root", type=str, help="Path to scene root containing sparse/0 and images/")
    parser.add_argument("--sparse_subdir", type=str, default="sparse/0", help="Relative path to COLMAP model")
    parser.add_argument("--near_perc", type=float, default=0.1, help="Near depth percentile")
    parser.add_argument("--far_perc", type=float, default=99.9, help="Far depth percentile")
    parser.add_argument("--out", type=str, default="poses_bounds.npy", help="Output filename (under scene_root)")
    args = parser.parse_args()

    scene_root = Path(args.scene_root).resolve()
    sparse_dir = (scene_root / args.sparse_subdir).resolve()
    out_path = scene_root / args.out

    if not sparse_dir.exists():
        raise FileNotFoundError(f"COLMAP model directory not found: {sparse_dir}")

    # pycolmap can read a directory containing cameras.*, images.*, points3D.*
    try:
        recon = pycolmap.Reconstruction(sparse_dir.as_posix())
    except RuntimeError as e:
        # Attempt to load text model if binary failed, and vice versa, by letting pycolmap probe.
        raise RuntimeError(
            f"Failed to read COLMAP model in {sparse_dir}. Ensure it contains cameras/ images/ points3D in "
            f".bin or .txt format. Original error: {e}"
        )

    if len(recon.images) == 0:
        raise RuntimeError("Reconstruction has zero registered images. Run COLMAP mapping first.")

    poses_bounds, names = build_poses_bounds(recon, args.near_perc, args.far_perc)
    np.save(out_path.as_posix(), poses_bounds)

    # Also write a companion list of image names for verification
    names_txt = scene_root / "image_names.txt"
    with open(names_txt, "w", encoding="utf-8") as f:
        for n in names:
            f.write(n + "\n")

    print(f"Wrote: {out_path}")
    print(f"Image order: {names_txt}")
    print(f"Shape: {poses_bounds.shape}")


if __name__ == "__main__":
    main()
