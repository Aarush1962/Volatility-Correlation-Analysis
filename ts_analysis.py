"""Volatility and correlation analysis of daily stock returns."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TICKERS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"]
START, END = "2021-01-01", "2024-12-31"
HORIZON = 5     # days ahead
ROLL_WIN = 20
LAMBDA = 0.94   # EWMA decay
USE_SYNTHETIC = False
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)


# data
def make_synthetic():
    rng = np.random.default_rng(0)
    n = 1000
    common = rng.normal(0, 1, n)
    cols = {}
    for tk in TICKERS:
        z = 0.6 * common + 0.8 * rng.normal(0, 1, n)
        vol, r = 0.01, []
        for zi in z:
            r.append(vol * zi)
            vol = np.sqrt(4e-6 + 0.1 * (vol * zi) ** 2 + 0.85 * vol ** 2)
        cols[tk] = 100 * np.exp(np.cumsum(r))
    return pd.DataFrame(cols, index=pd.bdate_range(START, periods=n))


def load_data():
    if USE_SYNTHETIC:
        print("USE_SYNTHETIC is True, using synthetic data")
        return make_synthetic()
    path = "data.csv"
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)
    try:
        import yfinance as yf
        df = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)["Close"]
        df = df[TICKERS].dropna()
        df.to_csv(path)
        return df
    except Exception as e:
        print("Download failed, using synthetic data:", e)
        return make_synthetic()


prices = load_data()
rets = np.log(prices).diff().dropna()
print(f"{len(rets)} daily returns, {rets.index[0].date()} to {rets.index[-1].date()}")
print("\nAnnualised volatility:\n", (rets.std() * np.sqrt(252)).round(3).to_string())


# 1. acf
def acf(x, nlags=20):
    x = np.asarray(x) - np.mean(x)
    return np.array([np.sum(x[k:] * x[:-k]) / np.sum(x ** 2) for k in range(1, nlags + 1)])


band = 1.96 / np.sqrt(len(rets))
rows = []
for tk in TICKERS:
    r = rets[tk].values
    rows.append((tk, acf(r, 1)[0], acf(r ** 2, 1)[0], np.mean(np.abs(acf(r ** 2)) > band)))
tab = pd.DataFrame(rows, columns=["ticker", "ACF1 returns", "ACF1 sq. returns", "share of 20 lags significant (sq.)"])
print("\nAutocorrelation summary (95%% band = +/-%.3f):\n" % band, tab.round(3).to_string(index=False))

first = TICKERS[0]
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
for a, vals, title in zip(ax, [acf(rets[first].values), acf(rets[first].values ** 2)],
                          [f"{first}: ACF of returns", f"{first}: ACF of squared returns"]):
    a.bar(range(1, 21), vals)
    a.axhline(band, color="r", ls="--")
    a.axhline(-band, color="r", ls="--")
    a.set_title(title)
plt.tight_layout()
plt.savefig(f"{OUT}/1_vol_clustering.png", dpi=120)
plt.close()


# 2. vol forecasts
def vol_forecasts(r):
    sq = r ** 2
    return pd.DataFrame({
        "rolling_20": np.sqrt(sq.rolling(ROLL_WIN).mean()),
        "ewma": np.sqrt(sq.ewm(alpha=1 - LAMBDA, adjust=False).mean()),
        "long_run": np.sqrt(sq.expanding(60).mean()),
        "target": np.sqrt(sq.rolling(HORIZON).mean().shift(-HORIZON)),
    })


errors = {m: [] for m in ["rolling_20", "ewma", "long_run"]}
for tk in TICKERS:
    d = vol_forecasts(rets[tk]).dropna()
    for m in errors:
        errors[m].extend((d[m] - d["target"]).values)

print("\nNext-%d-day volatility forecast RMSE (daily vol units, lower = better):" % HORIZON)
for m, e in errors.items():
    print("  %-11s %.5f" % (m, np.sqrt(np.mean(np.square(e)))))

d = vol_forecasts(rets[first]).dropna()
plt.figure(figsize=(10, 3.5))
plt.plot(d.index, d["target"], label="Realised next-5d vol", lw=1)
plt.plot(d.index, d["rolling_20"], label="Rolling 20d", lw=1)
plt.plot(d.index, d["ewma"], label=f"EWMA (lambda={LAMBDA})", lw=1)
plt.title(f"{first}: volatility forecasts vs realised")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/2_vol_forecast.png", dpi=120)
plt.close()


# 3. correlation
A, B = TICKERS[0], TICKERS[1]
roll_corr = rets[A].rolling(60).corr(rets[B])
print("\n%s vs %s: full-sample corr %.3f | rolling-60d range [%.2f, %.2f]"
      % (A, B, rets[A].corr(rets[B]), roll_corr.min(), roll_corr.max()))

lags = range(-5, 6)   # k > 0: A leads B
xcorr = [rets[A].corr(rets[B].shift(-k)) for k in lags]
print("Lead-lag corr(A_t, B_t+k):", {k: round(float(v), 3) for k, v in zip(lags, xcorr)})

fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax[0].plot(roll_corr.index, roll_corr.values)
ax[0].set_title(f"Rolling 60d correlation: {A} vs {B}")
ax[1].bar(list(lags), xcorr)
ax[1].set_title("Lead-lag cross-correlation (k>0: A leads B)")
plt.tight_layout()
plt.savefig(f"{OUT}/3_correlation_leadlag.png", dpi=120)
plt.close()

print(f"\nPlots saved in ./{OUT}/")
