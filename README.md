# Volatility & Correlation Analysis of Stock Returns

A small time series project on the risk and co-movement of stock returns, written with NumPy, pandas and Matplotlib. yfinance is used only to download the data.

## Questions
1. Do returns or squared returns show autocorrelation? (volatility clustering)
2. Which simple method best forecasts next-week volatility: rolling 20-day, EWMA, or the long-run average?
3. How stable is the correlation between two stocks, and does one lead the other?

## Data
- 4 large-cap Indian stocks: RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS
- Daily adjusted closes from 2021-01-01 to 2024-12-31, downloaded with yfinance
- Daily log returns
- The download is cached in `data.csv` (git-ignored); delete it to download again
- If the download fails, the script falls back to synthetic GARCH-like data and prints a message, so results from that run are not real market results
- Set `USE_SYNTHETIC = True` at the top of the script to skip `data.csv` and the download and use the synthetic data directly

## Method
1. ACF (lags 1-20) of returns and squared returns, with a 95% band of ±1.96/√N.
2. Forecast the realised volatility of the next 5 days using only data up to day t, with three methods: rolling 20-day, EWMA (λ = 0.94) and an expanding long-run average (at least 60 observations). The target is the root mean square return over days t+1 to t+5. Methods are compared by RMSE, pooled over all 4 stocks, in daily volatility units.
3. Full-sample and rolling 60-day correlation for RELIANCE vs TCS, plus the cross-correlation corr(A_t, B_t+k) for k = -5 to 5. A positive k means A leads B.

## Files
```
ts_analysis.py     # the whole analysis
requirements.txt
outputs/           # saved plots (1_vol_clustering, 2_vol_forecast, 3_correlation_leadlag)
```

## Run
```
pip install -r requirements.txt
python ts_analysis.py
```
The script prints annualised volatility per stock, an autocorrelation summary, the forecast RMSEs and the correlation numbers, and saves the plots to `outputs/`.

## Findings
Numbers are read off the saved plots, so they are approximate.

- Returns: lag-1 ACF is about 0.01. Only lags 17 and 18 slightly cross the ±0.062 band, which is within what chance would produce.
- Squared returns: lag-1 ACF is about 0.14 and 19 of 20 lags are above the band (lag 14 is the exception). The autocorrelation decays slowly, so volatility clusters.
- Forecasts: rolling 20-day and EWMA both follow realised volatility fairly well, including the mid-2023 spike, but react late to sudden jumps. RMSE: rolling_20 ... | EWMA ... | long_run ... -> best: ...
- RELIANCE vs TCS: full-sample correlation is about 0.35. The rolling 60-day value mostly stays between 0.2 and 0.55 and drops to about -0.2 in early 2024.
- Lead-lag: the correlation sits at k = 0 (about 0.35). Every other lag is below about 0.06, so there is no useful lead-lag at daily frequency.

## Limitations / next steps
- Only 4 stocks, one pair for the correlation analysis, and simple volatility models
- The 5-day targets overlap, so forecast errors are autocorrelated and small RMSE differences may not mean much
- Next: GARCH(1,1), more stocks, intraday data
