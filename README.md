# Forex Currency Forecasting

An end-to-end machine learning project: collect daily exchange rates from an OANDA rates API, compare seven forecasting approaches **per currency**, save the best deployable model for each, and serve forecasts in a Streamlit app packaged with Docker.

> **Disclaimer.** This is a learning and portfolio project. Exchange rates are close to a random walk, and the forecasts here are illustrative, **not financial advice**.

## What it does

- **Data:** 365 days (2025-04-17 to 2026-04-16) of daily rates for 11 currencies, quoted as units per 1 USD: GBP, JPY, AUD, INR, PKR, QAR, KWD, CNY, RUB, EUR, CAD.
- **Analysis:** cleaning, rebased trend plots, return correlations, volatility ranking and ADF stationarity tests.
- **Modeling:** naive baseline, ARIMA, SARIMA, Prophet, XGBoost, LightGBM and LSTM, each trained and evaluated on a chronological 60-day hold-out for every currency.
- **App:** pick a currency and a horizon (1 to 365 days) and get a forecast table and chart from that currency's saved model.
- **Deployment:** Dockerfile for one-command runs.

## Results

Best **deployable** model per currency, chosen by lowest RMSE on the last 60 days (train: 305 days, test: 60 days):

| Currency | Deployed model | MAPE (%) | RMSE | Naive RMSE | RMSE improvement vs naive |
|---|---|---|---|---|---|
| AUD | SARIMA | 0.93 | 0.0187 | 0.0189 | ~1% |
| CAD | XGBoost | 0.73 | 0.0124 | 0.0167 | ~26% |
| CNY | XGBoost | 0.40 | 0.0341 | 0.0379 | ~10% |
| EUR | LightGBM | 1.47 | 0.0153 | 0.0191 | ~20% |
| GBP | LightGBM | 1.16 | 0.0095 | 0.016 | ~41% |
| INR | Prophet | 1.03 | 1.1903 | 2.2244 | ~46% |
| JPY | LightGBM | 1.69 | 2.9522 | 5.6622 | ~48% |
| KWD | LightGBM | 0.28 | 0.0011 | 0.0014 | ~18% |
| PKR | Prophet | 0.06 | 0.2314 | 0.4665 | ~50% |
| QAR | SARIMA | 0.08 | 0.0039 | 0.0054 | ~28% |
| RUB | XGBoost | 2.30 | 2.5706 | 3.3781 | ~24% |

Where the LSTM scored best overall (CAD, JPY), the app deploys the best of the other models, because serving an LSTM would add TensorFlow to the image. The full model-by-model comparison is in the notebook.

### Key findings

- **No single model wins everywhere.** Prophet suits steadily drifting INR and PKR, SARIMA suits AUD and QAR, LightGBM suits GBP, EUR, KWD and JPY, and XGBoost suits CAD, CNY and RUB.
- **ARIMA is almost identical to the naive baseline**, as expected for near-random-walk data.
- **Prophet is the most uneven:** best for INR and PKR, but about 3 times worse than naive on AUD.
- **Related currencies move together.** EUR-GBP return correlation is 0.85, and PKR-QAR is 0.63. RUB and INR are largely independent of the rest.
- **Stationarity:** levels are non-stationary for most currencies, returns are stationary for all, which is why the tree models and LSTM predict returns.

<p>
<img src="docs/return_correlation.png" width="48%" alt="Return correlation heatmap">
<img src="docs/all_models_vs_actual_eur.png" width="48%" alt="All models vs actual for EUR">
</p>

## Limitations (read before trusting the numbers)

1. **Small sample.** One year of data and a single 60-day test window, so winners could change on another window.
2. **Selection bias.** The same window picks the model and reports its score, so the errors are optimistic. Rolling-origin cross-validation is the proper fix.
3. **The naive baseline is flat.** Where a currency trended in the test window (JPY about +3.7%, INR about +2.7%), a flat line loses badly, so large gains reflect trend capture in that window rather than proven skill.
4. **Recursive multi-step forecasts** compound error over long horizons.
5. **Hyperparameters are not tuned**, and the ARIMA order is fixed at (1,1,1).
6. **The data stops in April 2026**, so forecasts start from there.

## Project structure

```
.
├── analysis.ipynb                 # EDA, model comparison, model saving (start here)
├── app.py                         # Streamlit app
├── Dockerfile / .dockerignore
├── requirements.txt               # app + Docker dependencies
├── requirements-notebook.txt      # adds Jupyter, plotting, TensorFlow for the notebook
├── data/
│   ├── oanda_forex_timeseries.csv # raw download from the API
│   └── forex_data.csv             # cleaned data used by the app
├── models/                        # one model per currency + results.csv (written by the notebook)
└── docs/                          # figures used in this README
```

## Quickstart

### 1. Run the notebook

```bash
git clone <your-repo-url> && cd <repo-folder>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab analysis.ipynb
```

Run all cells. The final cells write `models/*.pkl`, `models/results.csv` and `data/forex_data.csv`. The dataset is already in the repo, so no API key is needed.

**To re-download fresh data** (optional): set an API key, delete the raw CSV, and re-run the first cell.

```bash
export OANDA_FX_KEY="your-key"
rm data/oanda_forex_timeseries.csv
```

Never commit API keys. The notebook reads the key only from the environment.

### 2. Run the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501.

### 3. Run with Docker

```bash
docker build -t forex-app .
docker run -p 8501:8501 forex-app
```

The image copies `app.py`, `data/` and `models/`, so run the notebook first and commit the generated `models/` files (or build locally after generating them).

> **Version note:** saved models are pickles, so the library versions in the Docker image must match the ones used for training. Pin exact versions in `requirements.txt` (for example from `pip freeze`). Only load pickles you created yourself.

## Method summary

- **Frequency:** calendar-daily (the data includes weekends), forward-filled as a safeguard.
- **Split:** chronological, last 60 days as test, never shuffled.
- **Metrics:** MAE, RMSE (used for selection) and MAPE (comparable across currencies).
- **Models:** ARIMA(1,1,1); SARIMA(1,1,1)(1,0,1,7); Prophet; XGBoost and LightGBM on 10 lagged returns plus day of week (recursive forecasting); LSTM on a 14-day window of standardized returns.
- **Saving:** each winner is refit on all data and saved as a small dict, so the app can load it without custom classes.

## Data and terms

Rates come from an OANDA rates API under a personal key limited to 2025-04-17 to 2026-04-17. Check the provider's terms before redistributing the raw data publicly.

## Roadmap

- Rolling-origin cross-validation and a separate validation window
- Prediction intervals in the app
- Longer history and scheduled retraining on fresh data
- Optional LSTM serving
