# MiLD — Synthetic unmixing (`code_MiLD`)

Minimal code for reproducing the **synthetic single-run** experiment (fixed loss weights, VCA-initialized decoders, `R1forMTHU2605110`).

## Layout

```
code_MiLD/
├── README.md
├── requirements.txt
├── train_syn1.py      # entry point (CLI)
├── model_syn1.py      # network
├── loss.py            # SAD, SparseKLloss, …
├── VCA.py             # VCA endmember extraction
└── data/              # create this; put your `.mat` here (see below)
    └── synth_dataset_ex1.mat
```

## Environment

```bash
cd code_MiLD
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install **PyTorch** matching your CUDA from [pytorch.org](https://pytorch.org) if you use GPU, then keep the rest from `requirements.txt`.

## Data

The training script expects a MATLAB file with at least:

| Variable | Description (this code) |
|----------|-------------------------|
| `Y`      | HSI tensor, reshaped to `L × H × W × T` with `L=224`, `H=W=50`, `T=6` |
| `A`      | Ground-truth abundances `P × H × W × T` with `P=3` |

Default path: **`data/synth_dataset_ex1.mat`** (relative to the `code_MiLD` directory). Copy your file there, or pass `--data /path/to/file.mat`.

## Run (default paths)

From inside `code_MiLD`:

```bash
python train_syn1.py
```

This uses:

- Data: `code_MiLD/data/synth_dataset_ex1.mat`
- Output: `code_MiLD/result/syn1/` (`.mat`, `images/*.png`, `run_summary.txt`)

## CLI options

```text
python train_syn1.py --help
```

Common overrides:

```bash
python train_syn1.py \
  --data /abs/path/synth_dataset_ex1.mat \
  --outdir /abs/path/my_results \
  --seed 8 \
  --epochs 100 \
  --a1 0.1 --a2 1.0 --a3 0.5 \
  --w-abu 2.0 --w-sto 0.05 --w-spk 0.001
```

- `--no-plots` — save `.mat` and summary only, skip figure export.
- `--detect-anomaly` — enable `torch.autograd.set_detect_anomaly(True)` (slower; for debugging).

## Reproducibility

- Random seed: `--seed` (default `8`).
- `torch.backends.cudnn.deterministic=True`; small numerical differences may still appear across CPU/GPU and library builds.

## Outputs

- `result/syn1/params_a1_0.1_a2_1_a3_0.5.mat` — estimated abundances, endmembers, reconstruction, hyperparameters, loss curve.
- `result/syn1/images/abundance_*_time*.png` — estimated vs. true abundances per time step.
- `result/syn1/run_summary.txt` — short run metadata.

## License

Add your license here if you distribute this bundle publicly.
