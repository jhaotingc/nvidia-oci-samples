# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import dataclass


@dataclass(frozen=True)
class FluxModelSpec:
    native_checkpoint: str
    t5_max_length: int
    default_steps: int
    default_guidance: float
    schedule_shift: bool


FLUX_MODEL_SPECS = {
    "flux-schnell": FluxModelSpec(
        native_checkpoint="flux1-schnell.safetensors",
        t5_max_length=256,
        default_steps=4,
        default_guidance=0.0,
        schedule_shift=False,
    ),
    "flux-dev": FluxModelSpec(
        native_checkpoint="flux1-dev.safetensors",
        t5_max_length=512,
        default_steps=50,
        default_guidance=3.5,
        schedule_shift=True,
    ),
}


def flux_model_spec(model_name: str) -> FluxModelSpec:
    try:
        return FLUX_MODEL_SPECS[model_name]
    except KeyError as exc:
        choices = ", ".join(FLUX_MODEL_SPECS)
        raise ValueError(f"Unknown FLUX model {model_name!r}; choose from {choices}") from exc
