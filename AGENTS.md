# Development

Run
`COMFYUI_ROOT=/absolute/ComfyUI COMFYUI_PYTHON=/absolute/prepared/python python3 scripts/setup_worktree.py`.
The explicit Python >=3.10 environment needs ComfyUI's torch, einops,
safetensors and pytest dependencies. Setup only checks prerequisites; it neither
installs packages nor loads models or connects to production. Rerunning is safe.

Run tests with `PYTHONPATH="$COMFYUI_ROOT" "$COMFYUI_PYTHON" -m pytest -q tests`.
GPU inference additionally requires separately provisioned compatible Krea 2,
Qwen3-VL and VAE weights. CPU fixtures do not establish GPU performance.
Shared operating guidance belongs to the existing `comfyui-model-runtimes` skill.
