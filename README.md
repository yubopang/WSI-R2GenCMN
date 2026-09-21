# WSI-R2GenCMN

WSI-R2GenCMN is a PyTorch implementation of a whole-slide image (WSI) report generation pipeline based on R2GenCMN. The code consumes pre-extracted WSI patch features (`.pt` files) and pathology report annotations, then trains a Transformer-style encoder-decoder with cross-modal memory to generate textual reports.

The repository also includes several multiple-instance learning (MIL) baseline modules and an OCR utility for extracting text from TCGA pathology report PDFs.

## Repository Structure

```text
.
├── main.py                  # Training and testing entry point
├── models/
│   ├── r2gencmn.py          # R2GenCMN model wrapper for feature inputs
│   └── r2gen.py             # Original R2Gen-style image model
├── modules/
│   ├── base_cmn.py          # Cross-modal memory Transformer implementation
│   ├── dataloaders.py       # Distributed data loaders and samplers
│   ├── datasets.py          # TCGA/WSI feature dataset
│   ├── tokenizers.py        # Report cleaning and vocabulary construction
│   ├── trainer.py           # Training, validation, testing, checkpointing
│   ├── metrics.py           # BLEU, METEOR, and ROUGE-L evaluation
│   └── ...
├── baselines/               # MIL baseline modules
├── ocr/
│   ├── pdf2text.py          # PDF-to-text OCR helper
│   └── dataset_csv/         # Example train/val/test split CSV files
└── LICENSE
```

## Requirements

This project is written for Python and PyTorch. The current entry point assumes CUDA and distributed training via NCCL.

Core dependencies:

```bash
pip install torch torchvision numpy pandas tqdm pillow opencv-python
pip install pycocoevalcap
pip install PyMuPDF pytesseract
pip install nystrom-attention
```

Notes:

- `pycocoevalcap` is used for BLEU, METEOR, and ROUGE-L.
- `pytesseract` requires the Tesseract OCR binary to be installed on the system.
- `nystrom-attention` is required by some baseline modules.
- The default code path uses `torch.distributed` with the `nccl` backend, so a CUDA-enabled environment is expected.

## Data Format

The main training code expects three inputs:

1. Patch feature directory, passed by `--image_dir`
2. Annotation directory, passed by `--ann_path`
3. Split CSV file, passed by `--split_path`

### Patch Features

`--image_dir` should contain one `.pt` file per slide:

```text
pt_files/
└── TCGA-XX-YYYY-01Z-00-DX1.uuid.pt
```

Each `.pt` file should load as a tensor of patch-level WSI features. By default, the code truncates each tensor to `--max_fea_length 10000`.

The default feature dimension is controlled by:

```bash
--d_vf 1024
```

Change this value if your extracted features have a different dimension.

### Annotations

`--ann_path` should contain one directory per TCGA case ID. Each case directory must include an `annotation` file containing a JSON string report:

```text
TCGA_BRCA/
└── TCGA-XX-YYYY/
    └── annotation
```

The tokenizer builds its vocabulary by scanning all `annotation` files under `--ann_path`.

### Splits

The split CSV should contain `train`, `val`, and `test` columns. Example files are provided in `ocr/dataset_csv/`.

```csv
train,val,test
TCGA-G8-6909-01Z-00-DX1...,TCGA-B6-A0IH-01Z-00-DX1...,TCGA-LD-A74U-01Z-00-DX1...
```

The dataset maps each slide name to a TCGA case ID using the first three dash-separated fields, for example:

```text
TCGA-G8-6909-01Z-00-DX1... -> TCGA-G8-6909
```

It then looks for:

```text
ann_path/TCGA-G8-6909/annotation
image_dir/TCGA-G8-6909-01Z-00-DX1....pt
```

## Training

Example single-node training command:

```bash
python main.py \
  --mode Train \
  --n_gpu 0 \
  --image_dir /path/to/pt_files \
  --ann_path /path/to/TCGA_BRCA \
  --split_path ocr/dataset_csv/splits_BRCA.csv \
  --save_dir results/BRCA \
  --record_dir records \
  --dataset_name tcga_organ \
  --batch_size 1 \
  --epochs 60
```


## Testing

To evaluate a saved checkpoint:

```bash
python main.py \
  --mode Test \
  --n_gpu 0 \
  --image_dir /path/to/pt_files \
  --ann_path /path/to/TCGA_BRCA \
  --split_path ocr/dataset_csv/splits_BRCA.csv \
  --checkpoint_dir results/BRCA \
  --save_dir results/BRCA \
  --dataset_name tcga_organ
```


## OCR Utility

`ocr/pdf2text.py` can be used to find TCGA PDF reports and diagnostic `.svs` files, convert PDF pages to images, and run OCR with Tesseract. This script is a data-preparation helper. The training pipeline itself expects cleaned `annotation` files and pre-extracted `.pt` WSI features.

