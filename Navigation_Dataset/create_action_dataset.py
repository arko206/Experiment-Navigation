#!/usr/bin/env python3

import math
from pathlib import Path


# ============================================================
# Dataset root
# ============================================================
DATASET_ROOT = Path.home() / "Navigation_Dataset"

CURRENT_POSE_DIR = DATASET_ROOT / "Present_Robot_Pose"
TARGET_POSE_DIR = DATASET_ROOT / "Target_Pose"
ACTION_DIR = DATASET_ROOT / "Action"

ACTION_DIR.mkdir(exist_ok=True)


# ============================================================
# Helper functions
# ============================================================
def wrap_angle(angle):
    """
    Wrap angle to [-pi, pi].
    """
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def read_pose_file(file_path):
    """
    Reads a pose file where each line is:
    x, y, theta
    """
    poses = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()

            if line == "":
                continue

            # Supports both comma-separated and space-separated values
            line = line.replace(",", " ")
            values = line.split()

            if len(values) != 3:
                raise ValueError(f"Invalid line in {file_path}: {line}")

            x, y, theta = map(float, values)
            poses.append((x, y, theta))

    return poses


def compute_robot_frame_action(curr_pose, target_pose):
    """
    Computes action from current pose to target pose.

    Input:
        curr_pose   = current world-frame pose
        target_pose = target world-frame pose

    Output:
        action = relative movement in current robot frame
               = dx_robot, dy_robot, dtheta
    """
    x_curr, y_curr, theta_curr = curr_pose
    x_tgt, y_tgt, theta_tgt = target_pose

    # Difference in world frame
    dx_world = x_tgt - x_curr
    dy_world = y_tgt - y_curr

    # Convert world difference into current robot frame
    dx_robot = math.cos(theta_curr) * dx_world + math.sin(theta_curr) * dy_world
    dy_robot = -math.sin(theta_curr) * dx_world + math.cos(theta_curr) * dy_world

    dtheta = wrap_angle(theta_tgt - theta_curr)

    return dx_robot, dy_robot, dtheta


# ============================================================
# Main processing
# ============================================================
def main():
    curr_files = sorted(CURRENT_POSE_DIR.glob("curr_pose_*.txt"))

    if len(curr_files) == 0:
        print(f"No curr_pose_*.txt files found in: {CURRENT_POSE_DIR}")
        return

    total_action_files = 0
    total_action_rows = 0

    for curr_file in curr_files:
        # Example:
        # curr_pose_1.txt -> index = 1
        index = curr_file.stem.replace("curr_pose_", "")

        target_file = TARGET_POSE_DIR / f"local_target_pose_{index}.txt"
        action_file = ACTION_DIR / f"action_{index}.txt"

        if not target_file.exists():
            print(f"[WARNING] Missing target file for {curr_file.name}: {target_file.name}")
            continue

        curr_poses = read_pose_file(curr_file)
        target_poses = read_pose_file(target_file)

        if len(curr_poses) != len(target_poses):
            raise ValueError(
                f"Row mismatch for index {index}: "
                f"{curr_file.name} has {len(curr_poses)} rows, "
                f"{target_file.name} has {len(target_poses)} rows."
            )

        actions = []

        for curr_pose, target_pose in zip(curr_poses, target_poses):
            action = compute_robot_frame_action(curr_pose, target_pose)
            actions.append(action)

        with open(action_file, "w") as f:
            for dx_robot, dy_robot, dtheta in actions:
                f.write(f"{dx_robot:.6f}, {dy_robot:.6f}, {dtheta:.6f}\n")

        print(f"[OK] Created {action_file.name} with {len(actions)} actions")

        total_action_files += 1
        total_action_rows += len(actions)

    print("\nDone.")
    print(f"Created action files: {total_action_files}")
    print(f"Total action rows: {total_action_rows}")
    print(f"Action folder: {ACTION_DIR}")


if __name__ == "__main__":
    main()

## Location for current pose: /home/unitree-arka/Navigation_Dataset/Present_Robot_Pose
## Location for target pose: /home/unitree-arka/Navigation_Dataset/Target_Pose
## Location for Local Obstacle Information: /home/unitree-arka/Navigation_Dataset/Local_Obstacle_Info
## Location for Action: /home/unitree-arka/Navigation_Dataset/Action
## Loation for Timestep: /home/unitree-arka/Navigation_Dataset/Timestep
