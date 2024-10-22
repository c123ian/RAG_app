# ---
# args: ["--force-download"]
# ---
import modal
app = modal.App("download_weights")

MODELS_DIR = "/llama_mini"  # create volume to store it in

DEFAULT_NAME = "meta-llama/Llama-3.2-3B"
# DEFAULT_REVISION = "8c22764a7e3675c50d4c7c9a4edb474456022b16"

volume = modal.Volume.from_name("llama_mini", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        [
            "huggingface_hub",  # download models from the Hugging Face Hub
            "hf-transfer",      # download models faster with Rust
            "transformers",      # for tokenizer
        ]
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


MINUTES = 60
HOURS = 60 * MINUTES

app = modal.App(image=image, secrets=[modal.Secret.from_name("huggingface")])

@app.function(volumes={MODELS_DIR: volume}, timeout=4 * HOURS)
def download_model(model_name, force_download=False):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    volume.reload()

    # Download the model weights and config
    snapshot_download(
        model_name,
        local_dir=MODELS_DIR,
        ignore_patterns=[
            "*.pt",
            "*.bin",
            "*.pth",
            "original/*",
        ],  # Ensure safetensors
        force_download=force_download,
    )

    # Download the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(MODELS_DIR)

    volume.commit()

@app.local_entrypoint()
def main(
    model_name: str = DEFAULT_NAME,
    force_download: bool = False,
):
    download_model.remote(model_name, force_download)
