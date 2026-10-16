"""Check the explicit ComfyUI checkout and runtime without starting inference."""
import os
from pathlib import Path
import subprocess
import sys

root = Path(os.environ.get("COMFYUI_ROOT", ""))
runtime = os.environ.get("COMFYUI_PYTHON")
if not root.is_absolute() or not (root / "comfy/ldm/krea2/model.py").is_file() or not runtime:
    sys.exit("Set COMFYUI_ROOT to the ComfyUI checkout and COMFYUI_PYTHON to its prepared Python >=3.10 runtime.")
subprocess.run([runtime, "-c", "import sys, torch, einops, safetensors; assert sys.version_info >= (3,10); print('Ready:', sys.version.split()[0], 'torch', torch.__version__)"], check=True)
print("No installs, model loads, service connections or tracked-file changes. Use COMFYUI_ROOT on PYTHONPATH for tests.")
