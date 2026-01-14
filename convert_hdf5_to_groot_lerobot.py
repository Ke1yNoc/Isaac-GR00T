"""
Script to convert UMI (Pika) HDF5 dataset to GR00T LeRobot format.

GR00T LeRobot is an enhanced version of LeRobot v2 with additional metadata:
- Parquet files with observation.state, action, timestamp, etc.
- MP4 video files in videos/chunk-XXX/
- meta/modality.json for state/action field definitions

This script generates the folder structure directly without using LeRobotDataset API.

Usage:
    python examples/umi/convert_hdf5_to_groot_lerobot.py \
        --data_dir /path/to/agilex/data \
        --output_dir /path/to/output/groot_dataset \
        --task_prompt "Fold the cloth."
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm
import tyro
from scipy.spatial.transform import Rotation


# =============================================================================
# ROBOT HOME TCP POSITION (computed from FK of home joint angles)
# From controllers.py: home_position = [0, 0.85, -0.59, 0.0, 0.65, 0.0]
# =============================================================================
ROBOT_HOME_TCP_RIGHT = np.array([0.27973, 0.0, 0.13828], dtype=np.float32)
ROBOT_HOME_TCP_LEFT = np.array([0.27973, 0.0, 0.13828], dtype=np.float32)


def compute_position_offset(
    episode_first_pos_right: np.ndarray,
    episode_first_pos_left: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the offset needed to transform episode positions to robot frame.
    
    offset = robot_home - episode_first_pos
    transformed_pos = original_pos + offset
    """
    offset_right = ROBOT_HOME_TCP_RIGHT - episode_first_pos_right
    offset_left = ROBOT_HOME_TCP_LEFT - episode_first_pos_left
    return offset_right, offset_left


def load_episode_data(
    episode_path: Path, 
    target_size: Optional[Tuple[int, int]] = None,
    apply_position_transform: bool = True,
) -> Optional[Dict]:
    """
    Load data from a single HDF5 episode.
    
    Returns dict with state, action, timestamps, and images.
    """
    try:
        with h5py.File(episode_path, "r") as f:
            # Check for required keys
            if 'localization/pose/pika_r' not in f:
                print(f"Skipping {episode_path}: Missing pika_r pose")
                return None

            # Load poses and gripper data
            pika_r_pose = f['localization/pose/pika_r'][:]  # (T, 6)
            pika_r_dist = f['gripper/encoderDistance/pika_r'][:].reshape(-1, 1)
            
            pika_l_pose = f['localization/pose/pika_l'][:]  # (T, 6)
            pika_l_dist = f['gripper/encoderDistance/pika_l'][:].reshape(-1, 1)
            
            # Load timestamps if available
            if 'localization/timestamp/pika_r' in f:
                timestamps = f['localization/timestamp/pika_r'][:]
            else:
                # Generate timestamps based on assumed 30fps
                timestamps = np.arange(len(pika_r_pose)) / 30.0
            
            # Ensure consistent length across all streams
            T = min(len(pika_r_pose), len(pika_l_pose), len(pika_r_dist), len(pika_l_dist))
            timestamps = timestamps[:T]
            
            # Extract positions
            robot0_eef_pos = pika_r_pose[:T, :3].astype(np.float32)
            robot1_eef_pos = pika_l_pose[:T, :3].astype(np.float32)
            
            # Apply position transformation
            if apply_position_transform:
                first_pos_right = robot0_eef_pos[0].copy()
                first_pos_left = robot1_eef_pos[0].copy()
                
                offset_right, offset_left = compute_position_offset(
                    first_pos_right, first_pos_left
                )
                
                robot0_eef_pos = robot0_eef_pos + offset_right
                robot1_eef_pos = robot1_eef_pos + offset_left
            
            # Convert Euler (Roll, Pitch, Yaw) to 6D rotation representation
            # Rot6D uses first two columns of rotation matrix for continuity
            robot0_rot_euler = pika_r_pose[:T, 3:6]
            robot0_rot_mat = Rotation.from_euler('xyz', robot0_rot_euler).as_matrix()  # (T, 3, 3)
            robot0_eef_rot = robot0_rot_mat[:, :, :2].reshape(-1, 6).astype(np.float32)  # (T, 6)
            robot0_gripper = pika_r_dist[:T].astype(np.float32)
            
            robot1_rot_euler = pika_l_pose[:T, 3:6]
            robot1_rot_mat = Rotation.from_euler('xyz', robot1_rot_euler).as_matrix()  # (T, 3, 3)
            robot1_eef_rot = robot1_rot_mat[:, :, :2].reshape(-1, 6).astype(np.float32)  # (T, 6)
            robot1_gripper = pika_l_dist[:T].astype(np.float32)
            
            # Construct 20D state vector
            # [r_pos(3), r_rot6d(6), r_grip(1), l_pos(3), l_rot6d(6), l_grip(1)]
            state = np.concatenate([
                robot0_eef_pos, robot0_eef_rot, robot0_gripper,
                robot1_eef_pos, robot1_eef_rot, robot1_gripper
            ], axis=1)

            # Construct Action: Store ABSOLUTE target positions (next state)
            # NOTE: For ActionRepresentation.RELATIVE in the config, the dataset
            # must store absolute values. The processor will handle the
            # absolute-to-relative conversion during training.
            action = np.zeros_like(state)
            action[:-1] = state[1:]  # Action at t = State at t+1
            action[-1] = state[-1]   # Repeat last state for final frame

            # Load images
            images_dict = {}
            camera_map = {
                'image_right': 'pikaFisheyeCamera_r',
                'image_left': 'pikaFisheyeCamera_l'
            }
            
            for key, cam_name in camera_map.items():
                ds_path = f'camera/color/{cam_name}'
                
                if ds_path in f:
                    paths = f[ds_path][:]
                    str_paths = [p.decode('utf-8') if isinstance(p, (bytes, np.bytes_)) else str(p) 
                                for p in paths]
                    str_paths = str_paths[:T]
                    
                    imgs = []
                    if target_size:
                        h_target, w_target = target_size
                    else:
                        h_target, w_target = 480, 640

                    for rel in str_paths:
                        img_path = (episode_path.parent / rel).resolve()
                        
                        if img_path.exists():
                            img = cv2.imread(str(img_path))
                            if img is not None:
                                # Resize if needed
                                if img.shape[:2] != (h_target, w_target):
                                    img = cv2.resize(img, (w_target, h_target))
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                imgs.append(img)
                            else:
                                imgs.append(np.zeros((h_target, w_target, 3), dtype=np.uint8))
                        else:
                            imgs.append(np.zeros((h_target, w_target, 3), dtype=np.uint8))
                    
                    images_dict[key] = np.stack(imgs, axis=0)

        return {
            "state": state,
            "action": action,
            "timestamps": timestamps,
            **images_dict
        }

    except Exception as e:
        print(f"Error loading {episode_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_episode_parquet(
    ep_data: Dict,
    episode_index: int,
    global_index_start: int,
    task_index: int,
) -> pd.DataFrame:
    """
    Create a DataFrame for one episode in GR00T LeRobot format.
    """
    num_frames = len(ep_data["state"])
    
    # Normalize timestamps to start from 0
    timestamps = ep_data["timestamps"]
    if len(timestamps) > 0:
        timestamps = timestamps - timestamps[0]
    
    records = []
    for i in range(num_frames):
        record = {
            "observation.state": ep_data["state"][i].tolist(),
            "action": ep_data["action"][i].tolist(),
            "timestamp": float(timestamps[i]) if i < len(timestamps) else i / 30.0,
            "task_index": task_index,
            "annotation.human.action.task_description": task_index,
            "annotation.human.validity": 1,  # All frames valid
            "episode_index": episode_index,
            "index": global_index_start + i,
            "next.reward": 0.0,
            "next.done": i == num_frames - 1,
        }
        records.append(record)
    
    return pd.DataFrame(records)


def create_video_from_images(
    images: np.ndarray,
    output_path: Path,
    fps: int = 30,
) -> bool:
    """
    Create MP4 video from numpy image array using OpenCV.
    
    Args:
        images: (T, H, W, 3) RGB images
        output_path: Path to output .mp4 file
        fps: Frames per second
    
    Returns:
        True if successful, False otherwise
    """
    if len(images) == 0:
        return False
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    h, w = images[0].shape[:2]
    
    # Use OpenCV VideoWriter with mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    if not writer.isOpened():
        print(f"Failed to open video writer for {output_path}")
        return False
    
    try:
        for img in images:
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            writer.write(img_bgr)
        return True
    finally:
        writer.release()


def generate_modality_json() -> Dict:
    """
    Generate the modality.json for 20D bimanual state/action.
    
    EEF format: XYZ position (3D) + rot6d (6D) = 9D per arm
    Total: [right_eef(9), right_gripper(1), left_eef(9), left_gripper(1)] = 20D
    """
    return {
        "state": {
            "right_eef": {"start": 0, "end": 9},
            "right_gripper": {"start": 9, "end": 10},
            "left_eef": {"start": 10, "end": 19},
            "left_gripper": {"start": 19, "end": 20}
        },
        "action": {
            "right_eef": {"start": 0, "end": 9},
            "right_gripper": {"start": 9, "end": 10},
            "left_eef": {"start": 10, "end": 19},
            "left_gripper": {"start": 19, "end": 20}
        },
        "video": {
            "right": {"original_key": "observation.images.right"},
            "left": {"original_key": "observation.images.left"}
        },
        "annotation": {
            "human.action.task_description": {"original_key": "annotation.human.action.task_description"},
            "human.validity": {"original_key": "annotation.human.validity"}
        }
    }


def generate_info_json(
    fps: int,
    robot_type: str,
    num_episodes: int,
    total_frames: int,
    image_size: Tuple[int, int],
    chunks_size: int = 1000,
) -> Dict:
    """
    Generate the info.json metadata file.
    """
    h, w = image_size
    return {
        "codebase_version": "v2.0",
        "robot_type": robot_type,
        "fps": fps,
        "total_episodes": num_episodes,
        "total_frames": total_frames,
        "chunks_size": chunks_size,
        "splits": {"train": f"0:{num_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [20],
                "names": {
                    "right_eef": [0, 9],
                    "right_gripper": [9, 10],
                    "left_eef": [10, 19],
                    "left_gripper": [19, 20]
                }
            },
            "action": {
                "dtype": "float32",
                "shape": [20],
                "names": {
                    "right_eef": [0, 9],
                    "right_gripper": [9, 10],
                    "left_eef": [10, 19],
                    "left_gripper": [19, 20]
                }
            },
            "observation.images.right": {
                "dtype": "video",
                "shape": [h, w, 3],
                "names": ["height", "width", "channel"],
                "video_info": {"fps": fps, "codec": "h264"}
            },
            "observation.images.left": {
                "dtype": "video",
                "shape": [h, w, 3],
                "names": ["height", "width", "channel"],
                "video_info": {"fps": fps, "codec": "h264"}
            }
        }
    }


def main(
    data_dir: Path,
    output_dir: Path,
    fps: int = 30,
    image_size: Tuple[int, int] = (480, 640),
    task_prompt: str = "Perform the manipulation task.",
    robot_type: str = "pika",
    apply_position_transform: bool = True,
):
    """
    Convert UMI HDF5 data to GR00T LeRobot format.
    
    Args:
        data_dir: Directory containing HDF5 episode files
        output_dir: Output directory for GR00T LeRobot dataset
        fps: Frames per second of the dataset
        image_size: Target image size (H, W)
        task_prompt: Task description to include
        robot_type: Robot type identifier
        apply_position_transform: Whether to apply coordinate transformation
    """
    if not data_dir.exists():
        raise ValueError(f"Dataset directory does not exist: {data_dir}")
    
    hdf5_files = sorted(data_dir.glob("**/data.hdf5"))
    print(f"Found {len(hdf5_files)} HDF5 episodes")
    
    if not hdf5_files:
        raise ValueError("No HDF5 files found in the dataset directory")

    # Clean up output directory
    output_dir = Path(output_dir)
    if output_dir.exists():
        print(f"Removing existing dataset at {output_dir}")
        shutil.rmtree(output_dir)
    
    # Create directory structure
    meta_dir = output_dir / "meta"
    data_dir_out = output_dir / "data" / "chunk-000"
    video_dir_right = output_dir / "videos" / "chunk-000" / "observation.images.right"
    video_dir_left = output_dir / "videos" / "chunk-000" / "observation.images.left"
    
    meta_dir.mkdir(parents=True, exist_ok=True)
    data_dir_out.mkdir(parents=True, exist_ok=True)
    video_dir_right.mkdir(parents=True, exist_ok=True)
    video_dir_left.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GR00T LEROBOT CONVERSION")
    print("=" * 60)
    if apply_position_transform:
        print(f"Position transformation: ENABLED")
        print(f"Robot home TCP (right): X={ROBOT_HOME_TCP_RIGHT[0]:.4f}, Y={ROBOT_HOME_TCP_RIGHT[1]:.4f}, Z={ROBOT_HOME_TCP_RIGHT[2]:.4f}")
    else:
        print(f"Position transformation: DISABLED")
    print("=" * 60)

    # Prepare tasks.jsonl
    tasks = [
        {"task_index": 0, "task": task_prompt},
        {"task_index": 1, "task": "valid"}
    ]
    
    # Process episodes
    episodes_info = []
    global_index = 0
    h, w = image_size
    
    for ep_idx, file_path in enumerate(tqdm.tqdm(hdf5_files, desc="Processing episodes")):
        ep_data = load_episode_data(
            file_path, 
            target_size=image_size,
            apply_position_transform=apply_position_transform
        )
        
        if ep_data is None:
            continue
        
        num_frames = len(ep_data["state"])
        
        # Create parquet file
        df = create_episode_parquet(
            ep_data,
            episode_index=ep_idx,
            global_index_start=global_index,
            task_index=0,
        )
        
        parquet_path = data_dir_out / f"episode_{ep_idx:06d}.parquet"
        df.to_parquet(parquet_path, index=False)
        
        # Create video files
        if "image_right" in ep_data and len(ep_data["image_right"]) > 0:
            video_path_right = video_dir_right / f"episode_{ep_idx:06d}.mp4"
            create_video_from_images(ep_data["image_right"], video_path_right, fps=fps)
        
        if "image_left" in ep_data and len(ep_data["image_left"]) > 0:
            video_path_left = video_dir_left / f"episode_{ep_idx:06d}.mp4"
            create_video_from_images(ep_data["image_left"], video_path_left, fps=fps)
        
        # Record episode info
        episodes_info.append({
            "episode_index": ep_idx,
            "tasks": [task_prompt],
            "length": num_frames
        })
        
        global_index += num_frames

    # Write meta files
    print("\nWriting meta files...")
    
    # tasks.jsonl
    with open(meta_dir / "tasks.jsonl", "w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")
    
    # episodes.jsonl
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep_info in episodes_info:
            f.write(json.dumps(ep_info) + "\n")
    
    # info.json
    info = generate_info_json(
        fps=fps,
        robot_type=robot_type,
        num_episodes=len(episodes_info),
        total_frames=global_index,
        image_size=image_size,
    )
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    # modality.json (GR00T-specific)
    modality = generate_modality_json()
    with open(meta_dir / "modality.json", "w") as f:
        json.dump(modality, f, indent=2)

    print("=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Total episodes: {len(episodes_info)}")
    print(f"Total frames: {global_index}")
    print(f"\nGenerated files:")
    print(f"  meta/info.json")
    print(f"  meta/tasks.jsonl")
    print(f"  meta/episodes.jsonl")
    print(f"  meta/modality.json")
    print(f"  data/chunk-000/episode_*.parquet ({len(episodes_info)} files)")
    print(f"  videos/chunk-000/observation.images.right/episode_*.mp4")
    print(f"  videos/chunk-000/observation.images.left/episode_*.mp4")
    print("=" * 60)


if __name__ == "__main__":
    tyro.cli(main)
