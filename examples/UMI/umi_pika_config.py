"""
UMI (Pika) bimanual robot configuration for GR00T finetuning.

This configuration defines how the UMI dataset should be loaded and processed:
- 20D state/action: [right_eef(9), right_gripper(1), left_eef(9), left_gripper(1)]
- EEF format: XYZ position (3D) + rot6d (6D) = 9D per arm
- Two fisheye cameras: right and left

The 6D rotation uses the first two columns of the rotation matrix for continuity:
    rot_mat = Rotation.from_euler('xyz', euler).as_matrix()
    rot6d = rot_mat[:, :2].flatten()  # 6D

Usage:
    python scripts/finetune.py --modality_config_path examples/UMI/umi_pika_config.py ...
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


umi_pika_config = {
    # Video modality: two fisheye cameras (right and left)
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["right", "left"],
    ),
    
    # State modality: 20D bimanual state
    # right_eef: XYZ pos (3) + rot6d (6) = 9D
    # right_gripper: 1D
    # left_eef: XYZ pos (3) + rot6d (6) = 9D  
    # left_gripper: 1D
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "right_eef",      # XYZ + Rot6D (9D)
            "right_gripper",  # Gripper distance (1D)
            "left_eef",       # XYZ + Rot6D (9D)
            "left_gripper",   # Gripper distance (1D)
        ],
    ),
    
    # Action modality: 16-step prediction horizon
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=[
            "right_eef",
            "right_gripper",
            "left_eef",
            "left_gripper",
        ],
        action_configs=[
            # Right arm EEF: relative control, XYZ + Rot6D format
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
            ),
            # Right gripper: absolute control
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # Left arm EEF: relative control, XYZ + Rot6D format
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
            ),
            # Left gripper: absolute control
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    
    # Language modality: task description annotation
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(umi_pika_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
