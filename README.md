# Fine-tuning AI Model with AMD Radeon RX 7800 XT

This is a learning project to explore supervised fine-tuning (SFT) of the Qwen3-0.6B AI model using an AMD Radeon RX 7800 XT GPU.
The project uses a custom question and answer dataset to fine-tune the model locally with Python, Hugging Face Transformers, and TRL.

This repository is intended for learning and experimentation, rather than production use.

## Prerequisite
```
uv
```
## Clone this repo
```
git clone https://github.com/tobing/finetune-ai-model-with-amd-7800xt.git
cd finetune-ai-model-with-amd-7800xt
```
## Prepare required python packages
> [!NOTE]
> Need to download more than 6 GB packages, as torch itself around 5.8 GB.
```
uv sync
uv pip install accelerate trl --no-deps # without --no-deps will install nvidia packages
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
```
## Prepare dataset
Modify file [`mydataset.csv`](mydataset.csv).
## Start finetuning the model
```
env -u PYTHONPATH -u PYTHONHOME -u PYTHON TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 uv run finetune
```
## Test the result
Before run the test, try to modify the prompt at [`test.py`](https://github.com/tobing/finetune-ai-model-with-amd-7800xt/blob/fa31fb540df13da92663cb281cea7aaa7bb4de2e/test.py#L17) based on your dataset
```
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 .venv/bin/python test.py
```


https://github.com/user-attachments/assets/623ab98f-1629-40f3-be34-c80ad9d0654c

