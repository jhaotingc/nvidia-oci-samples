# FLUX.1 inference benchmarking

Minimal `1024x1024` benchmarks for `FLUX.1-schnell` across BFL PyTorch, Hugging Face Diffusers, TorchAO, SGLang, vLLM Omni, TensorRT-LLM VisualGen, and TensorRT. Only optimized execution modes are retained. The advanced BFL PyTorch/TensorRT sweep also supports `FLUX.1-dev` with its guided, shifted denoising schedule; the reference comparison contract remains Schnell-specific.

## Measurement

Comparable runs use:

- Four denoising steps and guidance `0.0`.
- True request batching: `B` prompts -> `B` latents -> `B` VAE-decoded images.
- Timing from T5/CLIP encoding through completion of the decoded GPU tensor.
- CUDA synchronization before stopping the timer.
- No model loading, engine building, GPU-to-CPU copy, PIL conversion, image encoding, or file I/O in the timing window.

The SGLang and vLLM Omni runners use their public in-process APIs; they do not start HTTP servers or install worker hooks. Their timer covers the offline API call until all `B` outputs return, including local scheduling and batching overhead. Because that scope differs from the synchronized internal-runtime timer used by the other backends, offline API results are reported separately.

The complete contract is in [`configs/flux1-schnell-1024.json`](configs/flux1-schnell-1024.json).

## Quick start

Run each backend inside its tested container. Install an overlay only where the table says it is required.

| Backend | Container | Setup |
|---|---|---|
| BFL PyTorch, TensorRT | `nvcr.io/nvidia/pytorch:26.07-py3` | [Install the BFL overlay](INSTALL.md#bfl-pytorch) |
| TorchAO | `nvcr.io/nvidia/pytorch:26.07-py3` | [Install the TorchAO overlay](INSTALL.md#torchao) |
| HF Diffusers | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc22` | Dependencies are included; run the [Diffusers verification](INSTALL.md#hugging-face-diffusers) |
| TensorRT-LLM VisualGen | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc22` | Dependencies are included; run the [VisualGen verification](INSTALL.md#tensorrt-llm-visualgen) |
| SGLang | `lmsysorg/sglang:v0.5.12` | Use the official image and run the [SGLang verification](INSTALL.md#sglang) |
| vLLM Omni | `vllm/vllm-omni:v0.26.0` | Use the official image and run the [vLLM Omni verification](INSTALL.md#vllm-omni) |

For the minimal Diffusers example, start the TensorRT-LLM container shown above and verify that its preinstalled packages are available:

```bash
python3 -c 'import diffusers, torch; print(torch.__version__, diffusers.__version__)'
```

Then clone the parent samples repository inside the container or mount an existing checkout:

```bash
git clone https://github.com/NVIDIA/nvidia-oci-samples.git
cd nvidia-oci-samples/generative-ai-samples/flux-1-benchmarking
make test
python3 benchmark.py --list-modes
```

`make test` checks the repository utilities; it does not install an inference backend. Place the Diffusers-format checkpoint at the path passed to `--model`, then run the structural preflight without loading the model:

```bash
python3 benchmark.py \
  --mode hf-diffusers-bf16-compile \
  --model /models/FLUX.1-schnell-diffusers \
  --batches 1 \
  --warmup 1 \
  --iterations 2 \
  --output-dir results/smoke \
  --check-only
```

Remove `--check-only` to run the benchmark:

```bash
python3 benchmark.py \
  --mode hf-diffusers-bf16-compile \
  --model /models/FLUX.1-schnell-diffusers \
  --batches 1 \
  --warmup 1 \
  --iterations 2 \
  --output-dir results/smoke
```

For stable measurements, use `--warmup 2 --iterations 20` and batch sizes `1 2 4 8 16 32`.

The unified launcher selects the backend runner, VisualGen YAML, and backend-specific output directory. Use `--dry-run` to print the exact backend command. A failed batch writes its diagnostic JSON and returns a nonzero process status.

The example writes `results/smoke/hf-diffusers/hf-diffusers-bf16-compile/b1.json`. The runners record environment/load metadata; measurement runs that include B1 also save a sample for later inspection.

`--check-only` checks discoverable Python modules and expected checkpoint/plan files. It does not load the model, test GPU compatibility or memory capacity, or validate numerical/image quality.

## Advanced backend entry points

```text
flux_batch_sweep.py               BFL PyTorch and TensorRT
hf_diffusers_flux_sweep.py        Diffusers BF16 compile
torchao_diffusers_flux_sweep.py   TorchAO BF16/NVFP4
visualgen_flux_sweep.py           TensorRT-LLM VisualGen BF16/NVFP4
sglang_flux_sweep.py              SGLang DiffGenerator offline batching
vllm_omni_flux_sweep.py           vLLM Omni offline batching
```

All runners live under `benchmarks/flux1_schnell/`. Checkpoints and TensorRT plans stay outside this repository.

### FLUX.1-dev with a custom TensorRT ONNX transformer

The BFL/TensorRT sweep accepts `--model-name flux-schnell` (the default) or
`--model-name flux-dev`. Model-specific defaults are applied when `--steps` or
`--guidance` is omitted:

| Model name | Steps | Guidance | Schedule shift | T5 tokens |
|---|---:|---:|---|---:|
| `flux-schnell` | 4 | 0.0 | disabled | 256 |
| `flux-dev` | 50 | 3.5 | enabled | 512 |

An externally quantized transformer can be combined with the unmodified CLIP, T5, and VAE ONNX
components. Create a composed directory using links or copies:

```text
/models/FLUX.1-dev-onnx-composed/
  clip.opt/                 -> public FLUX.1-dev ONNX clip.opt/
  t5.opt/                   -> public FLUX.1-dev ONNX t5.opt/
  vae.opt/                  -> public FLUX.1-dev ONNX vae.opt/
  transformer.opt/fp4/      -> custom NVFP4 transformer ONNX directory
    model.onnx
    model.onnx_data
```

Build a static B1, 1024x1024 TensorRT plan set:

```bash
python3 -m benchmarks.flux1_schnell.flux_batch_sweep \
  --model-name flux-dev \
  --backend trt \
  --precision fp4 \
  --variant eager \
  --batch-sizes 1 \
  --onnx-dir /models/FLUX.1-dev-onnx-composed \
  --engine-root /engines/FLUX.1-dev-fp4-1024 \
  --output-dir results/flux-dev-build \
  --build-only
```

Generate one 28-step image with a caller-provided prompt:

```bash
python3 -m benchmarks.flux1_schnell.flux_batch_sweep \
  --model-name flux-dev \
  --backend trt \
  --precision fp4 \
  --variant eager \
  --batch-sizes 1 \
  --onnx-dir /models/FLUX.1-dev-onnx-composed \
  --engine-root /engines/FLUX.1-dev-fp4-1024 \
  --output-dir results/flux-dev-inference \
  --steps 28 \
  --guidance 3.5 \
  --warmup 0 \
  --iterations 1 \
  --batch-semantics images-per-prompt \
  --seed 42 \
  --prompt "A small red robot reading a book beneath a cedar tree, cinematic light"
```

The image is written below `results/flux-dev-inference/images/flux-dev/`, and the JSON under
`results/flux-dev-inference/runs/flux-dev/` records the resolved transformer ONNX and TensorRT
plan paths. Use `images-per-prompt` when the exact `--prompt` should be repeated across the batch;
`request-batch` uses the benchmark's canonical prompt bank.

FLUX.1-dev output is not comparable to the Schnell reference contract because its guidance and
denoising schedule differ.

See [`INSTALL.md`](INSTALL.md) for the installation and checkpoint setup for every backend.

See [`ENVIRONMENTS.md`](ENVIRONMENTS.md) for container provenance, observed package versions, hardware requirements, and the complete mode lookup table.

## Common setup errors

- `missing Python module`: use the container shown by `benchmark.py --list-modes`, then install the overlay linked in the Quick Start table.
- `missing ... checkpoint`: compare the supplied path with the layouts in [`INSTALL.md`](INSTALL.md#checkpoints).
- `missing TensorRT plan`: rerun the TensorRT mode with `--build-only` for the requested batch before measuring.
- CUDA Graph profiling: supported runners accept `--nsys-capture`; invoke Nsight Systems with `--cuda-graph-trace=node`.

## Nsight Systems profiling

`--nsys-capture` enables each runner's profiling boundary. Most runners use the
CUDA profiler API to capture measured work after warmup. The offline SGLang and
vLLM Omni runners instead annotate their requested warmup and measured API calls
with a `flux_offline_profile` NVTX range. Capture their full process and select
that range in Nsight to focus on the exact calls. The unified CLI executes the
backend in the profiler's target process; no `sitecustomize.py` or worker hook is
used. Run one batch size per offline profile.

`benchmarks.flux1_schnell.flux_t2i_trt11` also preserves the public BFL
`Fire(main)` entry point for direct invocation. For that compatibility path,
set `FLUX_NSYS_WARMUP_SAMPLES` and `FLUX_NSYS_MEASURED_SAMPLES` to positive
sample counts to start capture after the requested warmup saves and stop it
after the measured saves. The unified `benchmark.py` launcher does not use
those variables; it brackets its measured loop directly with the CUDA profiler API.

Example for SGLang B1 with two warmups and five measured calls:

```bash
nsys profile \
  --wait=primary \
  --trace=cuda,nvtx,cudnn,cublas \
  --cuda-trace-scope=process-tree \
  --cuda-graph-trace=node \
  --sample=none \
  --output=sglang-bf16-b1-w2-i5 \
  python3 benchmark.py \
    --mode sglang-bf16-offline-compile \
    --model /models/FLUX.1-schnell-diffusers \
    --batches 1 \
    --warmup 2 \
    --iterations 5 \
    --nsys-capture \
    --output-dir results/nsys
```

Replace the mode and output name for vLLM Omni or B4. `nsys` must be
available inside the selected backend container, either from the image or via
a compatible host bind mount. The profiled run includes collection
overhead; use normal multi-iteration runs for throughput reporting.

SGLang can leave its Nsight launcher attached after the worker reports
`Shutdown complete`. If that happens, press Ctrl-C once after the result JSON is
printed; Nsight finalizes and keeps the report. This is a profiler lifecycle
workaround only and does not add a worker hook.

## Slurm

The Slurm example has no embedded account, partition, storage path, or container:

```bash
sbatch \
  --partition=<gpu-partition> \
  --account=<account> \
  --export=ALL,MODEL_DIR=/models/FLUX.1-schnell-diffusers \
  scripts/slurm/smoke_hf_diffusers.sbatch
```

For this Diffusers smoke test, set `CONTAINER_IMAGE` to `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc22` or its site-local equivalent. Optional variables are `REPO_ROOT`, `OUTPUT_ROOT`, `CACHE_ROOT`, `PYTHON_BIN`, and `CONTAINER_MOUNTS`.

## Result contract check

`tools/check_result_contract.py` checks only whether result metadata follows the comparison contract:

- `1024x1024`, four steps, and true request-batch semantics.
- `B` canonical prompts produced `B` decoded images.
- `B` canonical prompt requests returned `B` outputs.
- Warmup and measured-iteration counts match the command.
- Latency and `images/s` are positive and mathematically consistent.

It does *not* validate image quality, numerical accuracy, checkpoint equivalence, or whether the implementation itself is correct. Those require separate output-quality and implementation reviews.

```bash
python3 -m tools.check_result_contract \
  --expected-batch 1 \
  --expected-warmup 1 \
  --expected-iterations 2 \
  results/smoke/hf-diffusers/hf-diffusers-bf16-compile/b1.json
```

For CUDA Graph profiling with Nsight Systems, use `--cuda-graph-trace=node`.

## License and contributions

This sample is part of [`NVIDIA/nvidia-oci-samples`](https://github.com/NVIDIA/nvidia-oci-samples) and is governed by its top-level [Apache License 2.0](https://github.com/NVIDIA/nvidia-oci-samples/blob/main/LICENSE). Adapted source and separately acquired runtime dependencies are identified in [Third-party notices](THIRD_PARTY_NOTICES.md).

Model checkpoints, compiled engines, Python environments, and container images are not distributed by this repository and remain subject to their publishers' terms.

Contributions are welcome under the parent repository's [contribution guidelines](https://github.com/NVIDIA/nvidia-oci-samples/blob/main/CONTRIBUTING.MD) and [CLA](https://github.com/NVIDIA/nvidia-oci-samples/blob/main/CLA.MD). Sign off every commit as required by that policy.
