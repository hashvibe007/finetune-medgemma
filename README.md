# Finetune Ayurlekha

A project for fine-tuning models on handwritten medical records dataset.

## Project Structure
```
.
├── main.py                 # Main script for dataset loading and processing
├── prescription-dataset/   # Directory for dataset files
├── requirements.txt        # Project dependencies
└── prompt_diary.md        # Development process documentation
```

## Setup

1. Create and activate virtual environment using uv :
```bash
uv init
 # On Linux/Mac
```

2. Install dependencies:
```bash
uv venv
uv add -r requirements.txt
```

## Usage

Run the main script:
```bash
uv run main.py
```

## Dataset

This project uses the "100-handwritten-medical-records" dataset from Hugging Face, which contains handwritten medical prescriptions.

## Development

For running this, we will use skypilot on vast ai for initial setup check and then on aws.
check documentation of skypilot to do setup
we need one yaml file e.g. finetune-medgemma.yaml 
<!-- For reference https://docs.skypilot.co/en/latest/getting-started/tutorial.html#ai-training -->


export HF_TOKEN=xxxx

uv run sky launch -c medgemma-prescription finetune_ayurlekha/finetune-medgemma-prod.yaml --env HF_TOKEN --env CHECKPOINT_BUCKET_NAME="medgemma-prescription"





