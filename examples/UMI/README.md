# UMI (Pika) Bimanual Robot Configuration

This directory contains the configuration files for finetuning GR00T with UMI Pika bimanual robot data.

## Data Format

The UMI dataset uses a **14-dimensional state/action space**:

| Field | Indices | Dimensions | Description |
|-------|---------|------------|-------------|
| `right_eef` | 0-6 | 6D | Right arm end-effector (XYZ position + rotation vector) |
| `right_gripper` | 6-7 | 1D | Right gripper distance |
| `left_eef` | 7-13 | 6D | Left arm end-effector (XYZ position + rotation vector) |
| `left_gripper` | 13-14 | 1D | Left gripper distance |

### Video

- `right`: Right fisheye camera (`pikaFisheyeCamera_r`)
- `left`: Left fisheye camera (`pikaFisheyeCamera_l`)

## Files

- `modality.json` - Defines slicing of state/action arrays and video key mappings
- `umi_pika_config.py` - Python configuration for model training

## Usage

### 1. Convert HDF5 to GR00T LeRobot format

```bash
python convert_hdf5_to_groot_lerobot.py \
    --data_dir /path/to/hdf5/episodes \
    --output_dir /path/to/groot_dataset \
    --task_prompt "Fold the cloth."
```

### 2. Copy modality.json to dataset

```bash
cp examples/UMI/modality.json /path/to/groot_dataset/meta/modality.json
```

### 3. Run finetuning

```bash
python scripts/finetune.py \
    --modality_config_path examples/UMI/umi_pika_config.py \
    --dataset_path /path/to/groot_dataset \
    ...
```

## Action Representation

- **EEF Control**: Uses `XYZ_ROTVEC` format (3D position + 3D rotation vector)
- **Relative Actions**: Arm movements are relative to current state
- **Absolute Grippers**: Gripper commands are absolute positions
- **Prediction Horizon**: 16 future timesteps
