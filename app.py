import joblib, numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st

st.set_page_config(page_title="Forex Forecaster", layout="wide")
DATA, LAGS = "data/forex_data.csv", 10
PEGGED = {"QAR", "KWD", "CNY"}

@st.cache_data
def load_data():
    df = pd.read_csv(DATA, index_col="Date", parse_dates=True).sort_index()
    return df[~df.index.duplicated()].asfreq("D").ffill()

@st.cache_resource
def load_model(pair):
    return joblib.load(f"models/{pair}_model.pkl")

@st.cache_data
def load_results():
    return pd.read_csv("models/results.csv")

def predict(a, h):
    """a = saved artifact: {kind, model, last_date, tail}. Must match the notebook."""
    idx = pd.date_range(a["last_date"] + pd.Timedelta(days=1), periods=h)
    k, last = a["kind"], a["tail"].iloc[-1]
    if k == "naive":
        v = np.repeat(last, h)
    elif k in ("arima", "sarima"):
        v = a["model"].forecast(h)
    elif k == "prophet":
        v = a["model"].predict(pd.DataFrame({"ds": idx}))["yhat"].values
    else:  # xgboost / lightgbm on lagged returns + day of week
        r, v, p = list(a["tail"].pct_change().dropna().values[-LAGS:]), [], last
        for d in idx:
            x = np.append(r[-LAGS:], d.dayofweek).reshape(1, -1)
            n = float(a["model"].predict(x)[0]); p *= 1 + n; v.append(p); r.append(n)
    return pd.Series(np.asarray(v, dtype=float), index=idx)

def make_forecast(pair, horizon):
    art = load_model(pair)  # loads the model chosen for this currency
    return art["kind"], predict(art, horizon)

st.title("💱 Forex Currency Forecaster")
data = load_data()
c1, c2 = st.columns(2)
pair = c1.selectbox("Currency (units per 1 USD)", list(data.columns))
horizon = int(c2.number_input("Forecast horizon (days)", 1, 365, 30))

kind, fc = make_forecast(pair, horizon)
hist = data[pair].iloc[-250:]
if pair in PEGGED:
    st.info(f"{pair} is pegged or tightly managed, so its forecast is expected to be nearly flat.")
fig = go.Figure([go.Scatter(x=hist.index, y=hist, name="History"),
                 go.Scatter(x=fc.index, y=fc, name="Forecast", line=dict(dash="dash"))])
fig.update_layout(title=f"{pair} — model: {kind}", height=450)
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)
left.subheader("Forecast"); left.dataframe(fc.rename("rate").round(5).to_frame(), use_container_width=True)
right.subheader("Model comparison (last 60 days)")
right.dataframe(load_results().query("pair == @pair").drop(columns="pair").round(5), use_container_width=True)
st.caption("Exchange rates are close to a random walk; treat forecasts as illustrative, not financial advice.")
