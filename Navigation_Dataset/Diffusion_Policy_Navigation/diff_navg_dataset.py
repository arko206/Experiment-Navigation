
#@markdown ### **Dataset**
#@markdown
#@markdown Defines `PushTStateDataset` and helper functions
#@markdown
#@markdown The dataset class
#@markdown - Load data (obs, action) from a zarr storage
#@markdown - Normalizes each dimension of obs and action to [-1,1]
#@markdown - Returns
#@markdown  - All possible segments with length `pred_horizon`
#@markdown  - Pads the beginning and the end of each episode with repetition
#@markdown  - key `obs`: shape (obs_horizon, obs_dim)
#@markdown  - key `action`: shape (pred_horizon, action_dim)
import numpy as np
import torch
import os
from pathlib import Path
import ast



def create_sample_indices(
        episode_ends:np.ndarray, sequence_length:int,
        pad_before: int=0, pad_after: int=0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result

# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    data_range = stats['max'] - stats['min']
    data_range[data_range < 1e-8] = 1.0

    # normalize to [0, 1]
    ndata = (data - stats['min']) / data_range

    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    data_range = stats['max'] - stats['min']
    data_range[data_range < 1e-8] = 1.0

    ndata = (ndata + 1) / 2
    data = ndata * data_range + stats['min']
    return data

def check_orientation_wrap(train_data, boundary_margin=0.0):
    """
    Checks whether world-frame orientations cross the -pi / +pi boundary.

    Observation layout:
        indices 0,1,2 = current x, y, theta
        indices 3,4,5 = target x, y, theta

    Action layout:
        indices 0,1,2 = delta_x, delta_y, delta_theta
    """

    obs = train_data["obs"]
    action = train_data["action"]

    current_theta = obs[:, 2]
    target_theta = obs[:, 5]
    action_delta_theta = action[:, 2]

    all_world_theta = np.concatenate(
        [current_theta, target_theta]
    )

    print("\n========== ORIENTATION CHECK ==========")

    print(
        "Current theta range:",
        float(current_theta.min()),
        "to",
        float(current_theta.max())
    )

    print(
        "Target theta range:",
        float(target_theta.min()),
        "to",
        float(target_theta.max())
    )

    print(
        "Action delta-theta range:",
        float(action_delta_theta.min()),
        "to",
        float(action_delta_theta.max())
    )

    # Check whether angles exist close to both +pi and -pi
    near_positive_pi = all_world_theta > (np.pi - boundary_margin)
    near_negative_pi = all_world_theta < (-np.pi + boundary_margin)

    # Check current-target pairs whose raw difference crosses the boundary
    boundary_crossings = np.abs(target_theta - current_theta) > np.pi

    # Check whether saved angles are outside the expected [-pi, pi] range
    world_angles_outside_range = (
        (all_world_theta < -np.pi) |
        (all_world_theta > np.pi)
    )

    action_angles_outside_range = (
        (action_delta_theta < -np.pi) |
        (action_delta_theta > np.pi)
    )

    if np.any(near_positive_pi) and np.any(near_negative_pi):
        print(
            "[WARNING] Dataset contains orientations near both "
            "+pi and -pi."
        )
        print(
            "Min-max normalization of world-frame theta may create "
            "an artificial discontinuity."
        )
    else:
        print(
            "[OK] Orientations are not concentrated near both "
            "+pi and -pi."
        )

    crossing_count = int(np.sum(boundary_crossings))

    print(
        "Current-to-target boundary crossings:",
        crossing_count
    )

    if crossing_count > 0:
        crossing_indices = np.where(boundary_crossings)[0]

        print(
            "[WARNING] Some current-target orientation pairs cross "
            "the wrap boundary."
        )

        print(
            "First crossing indices:",
            crossing_indices[:10]
        )

        for index in crossing_indices[:5]:
            print(
                f"  Row {index}: "
                f"current_theta={current_theta[index]:.6f}, "
                f"target_theta={target_theta[index]:.6f}, "
                f"action_delta_theta={action_delta_theta[index]:.6f}"
            )
    else:
        print("[OK] No current-target wrap-boundary crossing found.")

    if np.any(world_angles_outside_range):
        print(
            "[WARNING] Some current or target theta values are outside "
            "[-pi, pi]."
        )
    else:
        print("[OK] All current and target theta values are inside [-pi, pi].")

    if np.any(action_angles_outside_range):
        print(
            "[WARNING] Some action delta-theta values are outside "
            "[-pi, pi]."
        )
    else:
        print("[OK] All action delta-theta values are inside [-pi, pi].")

    print("=======================================\n")



##-- Added Helper Functions to read data from the saved text file ------####
def get_file_index(file_path, prefix):
    return int(file_path.stem.replace(prefix, ""))


def read_numeric_txt(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            line = line.replace(",", " ")
            values = [float(v) for v in line.split()]
            data.append(values)
    return np.array(data, dtype=np.float32)


def read_local_obstacle_txt(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue

            obs_list = ast.literal_eval(line)

            flat_obs = []
            for obs in obs_list:
                flat_obs.extend(obs)

            data.append(flat_obs)

    return np.array(data, dtype=np.float32)



##---- Added the Main Text-Dataset Loader ----------######################
def load_navigation_txt_dataset(dataset_root):
    dataset_root = Path(dataset_root)

    curr_pose_dir = dataset_root / "Present_Robot_Pose"
    target_pose_dir = dataset_root / "Target_Pose"
    local_obs_dir = dataset_root / "Local_Obstacle_Info"
    action_dir = dataset_root / "Action"

    curr_files = sorted(
        curr_pose_dir.glob("curr_pose_*.txt"),
        key=lambda p: get_file_index(p, "curr_pose_")
    )

    all_obs = []
    all_actions = []
    episode_ends = []

    total_count = 0

    for curr_file in curr_files:
        idx = get_file_index(curr_file, "curr_pose_")

        target_file = target_pose_dir / f"local_target_pose_{idx}.txt"
        obs_file = local_obs_dir / f"local_obs_{idx}.txt"
        action_file = action_dir / f"action_{idx}.txt"

        curr_pose = read_numeric_txt(curr_file)
        target_pose = read_numeric_txt(target_file)
        local_obs = read_local_obstacle_txt(obs_file)
        action = read_numeric_txt(action_file)

        if not (
            len(curr_pose) == len(target_pose) == len(local_obs) == len(action)
        ):
            raise ValueError(
                f"Length mismatch in sample {idx}: "
                f"curr={len(curr_pose)}, "
                f"target={len(target_pose)}, "
                f"obs={len(local_obs)}, "
                f"action={len(action)}"
            )

        obs = np.concatenate(
            [curr_pose, target_pose, local_obs],
            axis=1
        )

        all_obs.append(obs)
        all_actions.append(action)

        total_count += len(action)
        episode_ends.append(total_count)

    train_data = {
        "obs": np.concatenate(all_obs, axis=0),
        "action": np.concatenate(all_actions, axis=0)
    }

    episode_ends = np.array(episode_ends, dtype=np.int64)

    return train_data, episode_ends






# dataset
class Navigation_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path,
                 pred_horizon, obs_horizon, action_horizon):

        # read from text files
    
       
        train_data, episode_ends = load_navigation_txt_dataset(dataset_path)
        check_orientation_wrap(train_data)
        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            # add padding such that each timestep in the dataset are seen
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1)

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        # all possible segments of the dataset
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        # discard unused observations
        nsample['obs'] = nsample['obs'][:self.obs_horizon,:]
        return nsample
    
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                    default='/home/unitree-arka/Navigation_Dataset')
    
    parser.add_argument('--pred_horizon', type=int, default=1)
    parser.add_argument('--obs_horizon', type=int, default=1)
    parser.add_argument('--action_horizon', type=int, default=1)
    args = parser.parse_args()



    idx = 0



    
    # if you kept hardcoded paths above, replace them with args.dataset_path
    # and args.* for horizons before constructing PushTStateDataset
    dataset = Navigation_Dataset(
        dataset_path=args.dataset_path,
        pred_horizon=args.pred_horizon,
        obs_horizon=args.obs_horizon,
        action_horizon=args.action_horizon
    )

    stats = dataset.stats

    # … your verification / dataloader code here …
    batch = next(iter(torch.utils.data.DataLoader(
        dataset, batch_size=256, num_workers=1, shuffle=True,
        pin_memory=True, persistent_workers=True
    )))
    print("batch['obs'].shape:", batch['obs'].shape)
    print("batch['action'].shape:", batch['action'].shape)

    buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = dataset.indices[idx]

    print("The buffer start index is: ", buffer_start_idx)
    print("The buffer end index is: ", buffer_end_idx)
    print("The sample start index is: ", sample_start_idx)
    print("The sample end index is: ", sample_end_idx)
    print("The length of one index entry is: ", len(dataset.indices[idx]))
    print("The total number of dataset samples is: ", len(dataset))
    

    nsample_at_first_index = dataset.__getitem__(idx)

    print("The first sample of the dataset is: ", nsample_at_first_index)

    obs_norm = nsample_at_first_index['obs']

    print("The normalized observation at the first index is: ", obs_norm)
    print("The shape of the normalized observation at the first index is: ", obs_norm.shape)

    action_norm = nsample_at_first_index['action']
    print("The normalized action at the first index is: ", action_norm)
    print("The shape of the normalized action at the first index is: ", action_norm.shape)

    ###unormalizing the data
    obs = unnormalize_data(obs_norm, stats['obs'])
    print("The unnormalized observation at the first index is: ", obs)
    print("The shape of the unnormalized observation at the first index is: ", obs.shape)

    action = unnormalize_data(action_norm, stats['action'])
    print("The unnormalized action at the first index is: ", action)
    print("The shape of the unnormalized action at the first index is: ", action.shape)


 

if __name__ == '__main__':
    main()
