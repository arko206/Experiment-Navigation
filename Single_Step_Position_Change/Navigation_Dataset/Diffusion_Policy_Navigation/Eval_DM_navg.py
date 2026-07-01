#@markdown ### **Network**
#@markdown
#@markdown Defines a 1D UNet architecture `ConditionalUnet1D`
#@markdown as the noies prediction network
#@markdown
#@markdown Components
#@markdown - `SinusoidalPosEmb` Positional encoding for the diffusion iteration k
#@markdown - `Downsample1d` Strided convolution to reduce temporal resolution
#@markdown - `Upsample1d` Transposed convolution to increase temporal resolution
#@markdown - `Conv1dBlock` Conv1d --> GroupNorm --> Mish
#@markdown - `ConditionalResidualBlock1D` Takes two inputs `x` and `cond`. \
#@markdown `x` is passed through 2 `Conv1dBlock` stacked together with residual connection.
#@markdown `cond` is applied to `x` with [FiLM](https://arxiv.org/abs/1709.07871) conditioning.

import argparse
import math
import sys
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

##---- already function definitions are existed in the "diff_navg_dataset.py" script --------####
from diff_navg_dataset import  normalize_data, unnormalize_data


#@markdown ### **Network**
#@markdown
#@markdown Defines a 1D UNet architecture `ConditionalUnet1D`
#@markdown as the noies prediction network
#@markdown
#@markdown Components
#@markdown - `SinusoidalPosEmb` Positional encoding for the diffusion iteration k
#@markdown - `Downsample1d` Strided convolution to reduce temporal resolution
#@markdown - `Upsample1d` Transposed convolution to increase temporal resolution
#@markdown - `Conv1dBlock` Conv1d --> GroupNorm --> Mish
#@markdown - `ConditionalResidualBlock1D` Takes two inputs `x` and `cond`. \
#@markdown `x` is passed through 2 `Conv1dBlock` stacked together with residual connection.
#@markdown `cond` is applied to `x` with [FiLM](https://arxiv.org/abs/1709.07871) conditioning.

####----- Loading the Diffusion Policy Model-------####
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            cond_dim,
            kernel_size=3,
            n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:,0,...]
        bias = embed[:,1,...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=5,
        n_groups=8
        ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv
        ##-----##-----Print model parameter count to stderr-----###-----###
        print(
            "number of parameters: {:e}".format(
                sum(p.numel() for p in self.parameters())
            ),
            file=sys.stderr,
        )

    def forward(self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            global_cond=None):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1,-2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1,-2)
        # (B,T,C)
        return x


OBS_DIM = 12
ACTION_DIM = 3

OBS_HORIZON = 1
PRED_HORIZON = 1

NUM_DIFFUSION_ITERS = 200   




# ============================================================
# Navigation diffusion evaluator
# ============================================================

class NavigationDiffusionEvaluator:
    def __init__(
        self,
        checkpoint_path: Path,
        stats_path: Path,
        device: torch.device,
    ):
        self.device = device

        # This architecture must match the training script exactly.
        self.model = ConditionalUnet1D(
            input_dim=ACTION_DIM,
            global_cond_dim=OBS_DIM * OBS_HORIZON,
            down_dims=[256],
        ).to(device)

        state_dict = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )

        self.model.load_state_dict(
            state_dict,
            strict=True,
        )

        self.model.eval()

        # This scheduler must also match training exactly.
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=NUM_DIFFUSION_ITERS,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

        # Load the min/max values saved during training.
        loaded_stats = np.load(stats_path)

        self.obs_min = np.asarray(
            loaded_stats["obs_min"],
            dtype=np.float32,
        )

        self.obs_max = np.asarray(
            loaded_stats["obs_max"],
            dtype=np.float32,
        )

        self.action_min = np.asarray(
            loaded_stats["action_min"],
            dtype=np.float32,
        )

        self.action_max = np.asarray(
            loaded_stats["action_max"],
            dtype=np.float32,
        )
        
        ###--- Store statistics in dictionaries for easy access during normalization/unnormalization. ---###
        self.obs_stats = {
            "min": self.obs_min,
            "max": self.obs_max,
        }

        self.action_stats = {
            "min": self.action_min,
            "max": self.action_max,
        }

        # Validate saved statistics.
        if self.obs_min.shape != (OBS_DIM,):
            raise ValueError(
                f"Expected obs_min shape {(OBS_DIM,)}, "
                f"but received {self.obs_min.shape}"
            )

        if self.obs_max.shape != (OBS_DIM,):
            raise ValueError(
                f"Expected obs_max shape {(OBS_DIM,)}, "
                f"but received {self.obs_max.shape}"
            )

        if self.action_min.shape != (ACTION_DIM,):
            raise ValueError(
                f"Expected action_min shape {(ACTION_DIM,)}, "
                f"but received {self.action_min.shape}"
            )

        if self.action_max.shape != (ACTION_DIM,):
            raise ValueError(
                f"Expected action_max shape {(ACTION_DIM,)}, "
                f"but received {self.action_max.shape}"
            )
    ##-- Adding One Step Prediction Function --------------####
    def predict(self, observation: np.ndarray) -> np.ndarray:

        """One-step prediction.

        observation:
            Shape (12,)
        returns:
            Shape (3,) -> [delta_x_robot, delta_y_robot, delta_theta]
        """

        observation = np.asarray(
            observation,
            dtype=np.float32,
        )

        if observation.shape != (OBS_DIM,):
            raise ValueError(
                f"Expected observation shape {(OBS_DIM,)}, "
                f"but received {observation.shape}"
            )

        if not np.all(np.isfinite(observation)):
            raise ValueError(
                "Observation contains NaN or infinity."
            )

        # ----------------------------------------------------
        # Normalize one 12-dimensional observation
        # ----------------------------------------------------
        normalized_observation = normalize_data(
            observation,
            stats=self.obs_stats,
        ).astype(np.float32)

        # Shape:
        # (12,) -> (1, 12)
        obs_cond = torch.from_numpy(
            normalized_observation
        ).to(
            device=self.device,
            dtype=torch.float32,
        )

        obs_cond = obs_cond.unsqueeze(0)

        # ----------------------------------------------------
        # Reverse diffusion process
        # ----------------------------------------------------
        with torch.inference_mode():
            # Shape:
            # batch=1, prediction horizon=1, action dimension=3
            normalized_action = torch.randn(
                (
                    1,
                    PRED_HORIZON,
                    ACTION_DIM,
                ),
                device=self.device,
                dtype=torch.float32,
            )

            self.noise_scheduler.set_timesteps(
                NUM_DIFFUSION_ITERS
            )
            ##------ Denoising Function / Reverse Diffusion Process-------#####
            for timestep in self.noise_scheduler.timesteps:
                predicted_noise = self.model(
                    sample=normalized_action,
                    timestep=timestep,
                    global_cond=obs_cond,
                )

                normalized_action = (
                    self.noise_scheduler.step(
                        model_output=predicted_noise,
                        timestep=timestep,
                        sample=normalized_action,
                    ).prev_sample
                )

            # Shape:
            # (1, 1, 3) -> (3,)
            normalized_action_np = (
                normalized_action
                .detach()
                .cpu()
                .numpy()[0, 0]
                .astype(np.float32)
            )

            # Convert back to physical robot-frame deltas.
            predicted_action = unnormalize_data(
                normalized_action_np,
                stats=self.action_stats,
            ).astype(np.float32)

            if predicted_action.shape != (ACTION_DIM,):
                raise RuntimeError(
                    "Unexpected predicted action shape: "
                    f"{predicted_action.shape}"
                )

            if not np.all(np.isfinite(predicted_action)):
                raise RuntimeError(
                    "Diffusion model produced NaN or infinity."
                )

            return predicted_action

###--- Parsing Arguments-----###
def parse_arguments():

    diffusion_policy_directory = (
        Path.home()
        / "Single_Step_Position_Change"
        / "Navigation_Dataset"
        / "Diffusion_Policy_Navigation"
    )

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            diffusion_policy_directory
            / "checkpoints"
            / "diffusion_navigation_one_step.ckpt"
        ),
    )

    parser.add_argument(
        "--stats",
        type=Path,
        default=(
            diffusion_policy_directory
            / "navigation_normalization_stats.npz"
        ),
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()

def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if (
        device_name == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is unavailable."
        )

    return torch.device(device_name)


#####---- Adding thenew Main Function for the Diffusion Policy code -----#####
def main():

    args = parse_arguments()

    if not args.checkpoint.is_file():
        print(
            f"Checkpoint was not found: "
            f"{args.checkpoint}",
            file=sys.stderr,
            flush=True,
        )
        return 1

    if not args.stats.is_file():
        print(
            f"Normalization statistics were not found: "
            f"{args.stats}",
            file=sys.stderr,
            flush=True,
        )
        return 1

    device = resolve_device(args.device)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    ##--- Printing the supplied seed values----####
    print(
    f"Diffusion inference seed: {args.seed}",
    file=sys.stderr,
    flush=True,
    )

    print(
        f"Loading navigation diffusion model on {device}.",
        file=sys.stderr,
        flush=True,
    )

    evaluator = NavigationDiffusionEvaluator(
        checkpoint_path=args.checkpoint,
        stats_path=args.stats,
        device=device,
    )

    print(
        "Navigation diffusion server is ready.",
        file=sys.stderr,
        flush=True,
    )

    # This message is read by the C++ process.
    print("READY", flush=True)

    # Keep the model loaded and process repeated requests.
    for raw_line in sys.stdin:
        line = raw_line.strip()

        if not line:
            continue

        if line.upper() == "QUIT":
            break

        try:
            # Also accept comma-separated values.
            tokens = line.replace(",", " ").split()

            values = [
                float(token)
                for token in tokens
            ]

            if len(values) != OBS_DIM:
                raise ValueError(
                    f"Expected {OBS_DIM} observation values, "
                    f"but received {len(values)}."
                )

            observation = np.asarray(
                values,
                dtype=np.float32,
            )

            action = evaluator.predict(observation)

            # Stdout must contain only machine-readable values.
            print(
                f"{action[0]:.10g} "
                f"{action[1]:.10g} "
                f"{action[2]:.10g}",
                flush=True,
            )

        except Exception as error:
            print(
                f"Inference request failed: {error}",
                file=sys.stderr,
                flush=True,
            )

            # C++ will reject non-finite outputs.
            print(
                "nan nan nan",
                flush=True,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())























































  
    


































    





















    