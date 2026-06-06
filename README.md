# Martingale Difference Divergence (MDD) with Macro

Applies a kernel‑based test for the martingale difference property. The MDD statistic measures how much the conditional expectation of ETF returns given macro variables deviates from zero. High MDD indicates predictability – a potential alpha signal.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Uses all available macro variables (VIX, DXY, yields)
- Gaussian kernel on standardised macro features
- Score = MDD = Σ_{i≠j} K(macro_i, macro_j) * ret_i * ret_j
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-mdd-macro-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (fast, O(n²) per window)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High MDD → ETF returns are predictable from macro (non‑martingale).
- Low MDD → returns are close to a martingale (unpredictable).

## Requirements

See `requirements.txt`.
