uv init
uv python pin 3.14
uv add datasets transformers psutil
uv pip install accelerate trl --no-deps
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
env -u PYTHONPATH -u PYTHONHOME -u PYTHON TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 uv run finetune
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 .venv/bin/python test.py