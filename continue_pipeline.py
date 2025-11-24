#!/usr/bin/env python3
"""
Continue pipeline from your plot:
- compute features (returns, lags, SMA, RSI)
- define target: next-day log return
- time split, train RandomForest baseline
- evaluate MAE, RMSE, direction accuracy
- plot preds vs actual and do a simple backtest (with transaction cost)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from joblib import dump
import warnings
warnings.filterwarnings("ignore")

# --- Parameters you can tweak ---
TICKER = "AAPL"
START = "2020-01-01"
END = "2024-01-01"
SEQ = False   # not used here; placeholder if you want sequences later
TEST_FRAC = 0.2
RANDOM_SEED = 42
TRANSACTION_COST = 0.001  # 0.1% per trade (adjust for realism)

# --- 1) load data (use what you already have or download) ---
# If you already have `data` in memory from your session, use it.
try:
    data  # if variable exists in notebook
    print("Using existing `data` variable from notebook.")
except NameError:
    print("Downloading data with yfinance...")
    data = yf.download(TICKER, start=START, end=END, progress=False)

# Ensure we have the expected columns
df = data.copy()
df.index.name = "date"
# use adjusted close if available
df["close"] = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]

# --- 2) feature engineering ---
def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(period, min_periods=period).mean()
    ma_down = down.rolling(period, min_periods=period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

df["log_ret"] = np.log(df["close"]).diff()
# create lags of returns
for lag in range(1, 6):
    df[f"lag_ret_{lag}"] = df["log_ret"].shift(lag)
# moving averages (price)
for w in (5, 10, 20):
    df[f"sma_{w}"] = df["close"].rolling(w).mean()
# rolling std of returns
for w in (5, 10, 20):
    df[f"roll_std_{w}"] = df["log_ret"].rolling(w).std()
# RSI
df["rsi_14"] = compute_rsi(df["close"], 14)
# volume normalized
df["vol_mean_10"] = df["Volume"].rolling(10).mean()
df["vol_rel"] = df["Volume"] / df["vol_mean_10"]

# target: next-day log return
df["target_ret_1"] = df["log_ret"].shift(-1)

# drop rows with NaNs
df = df.dropna().copy()
print(f"Data after feature engineering: {df.shape[0]} rows, {df.shape[1]} columns")

# --- 3) prepare feature matrix and target ---
feature_cols = [c for c in df.columns if c.startswith("lag_ret_") or c.startswith("sma_") or c.startswith("roll_std_") or c in ("rsi_14", "vol_rel")]
X = df[feature_cols]
y = df["target_ret_1"]

# Optional: quick correlation check (helpful to see signal strength)
print("\nTop 5 absolute correlations with target:")
print(df[feature_cols + ["target_ret_1"]].corr()["target_ret_1"].abs().sort_values(ascending=False).head(6))

# --- 4) time split (train on first 80%, test on last 20%) ---
split_idx = int(len(df) * (1 - TEST_FRAC))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
print(f"\nTrain rows: {len(X_train)}, Test rows: {len(X_test)}")

# scale features (fit only on train)
scaler = StandardScaler().fit(X_train)
X_train_s = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=feature_cols)
X_test_s = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=feature_cols)

# --- 5) baselines: naive and simple models ---
# Naive baseline: predict 0 (no change) and previous-day return
naive_preds = np.zeros_like(y_test.values)
prevday_preds = X_test["lag_ret_1"].values  # predict yesterday's return

def direction_accuracy(y_true, y_pred):
    return (np.sign(y_true) == np.sign(y_pred)).mean()

def print_metrics(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    dir_acc = direction_accuracy(y_true, y_pred)
    print(f"{name:20s} MAE: {mae:.6f}, RMSE: {rmse:.6f}, DirAcc: {dir_acc:.3f}")

print("\nBaselines:")
print_metrics("Naive (0)", y_test, naive_preds)
print_metrics("PrevDay", y_test, prevday_preds)

# Ridge regression baseline
ridge = Ridge(alpha=1.0).fit(X_train_s, y_train)
ridge_preds = ridge.predict(X_test_s)
print_metrics("Ridge", y_test, ridge_preds)

# RandomForest baseline
rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
rf.fit(X_train, y_train)  # RF with raw features (trees don't need scaling)
rf_preds = rf.predict(X_test)
print_metrics("RandomForest", y_test, rf_preds)

# save model if you like
dump(rf, f"rf_{TICKER}.joblib")
print("\nSaved RandomForest model: ", f"rf_{TICKER}.joblib")

# --- 6) quick plots: predicted vs actual next-day return (test period) ---
plt.figure(figsize=(12,5))
plt.plot(y_test.index, y_test.values, label="actual next-day log return", alpha=0.7)
plt.plot(y_test.index, rf_preds, label="RF preds", alpha=0.7)
plt.legend()
plt.title(f"{TICKER} actual vs RF predicted next-day log return (test)")
plt.show()

# scatter
plt.figure(figsize=(5,5))
plt.scatter(y_test, rf_preds, alpha=0.4, s=10)
plt.xlabel("Actual next-day log return")
plt.ylabel("RF predicted")
plt.title("Scatter actual vs predicted")
plt.axline((0,0),(1,1), color="red", linewidth=0.8)
plt.show()

# feature importances (RF)
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 10 feature importances (RF):")
print(importances.head(10))

# --- 7) simple backtest: sign-based long/short strategy using predictions ---
# Position = sign(prediction): 1 for long, -1 for short
pos = np.sign(rf_preds)
# Align positions with actual next-day returns: we take position at close of day t to capture return on day t+1
pos_series = pd.Series(pos, index=y_test.index)
# compute strategy returns: position * actual next-day log return
strat_ret = pos_series * y_test
# compute transaction costs when position changes: cost proportional to absolute change in position
pos_prev = pos_series.shift(1).fillna(0)
trades = (pos_series != pos_prev).astype(int)
# cost per trade in log-return approximation
strat_ret_after_costs = strat_ret - trades * TRANSACTION_COST

# cumulative returns (log-returns -> cumulative product)
cum_strat = (strat_ret_after_costs).cumsum().apply(np.exp)
cum_hold = y_test.cumsum().apply(np.exp)  # buy-and-hold on next-day returns (for comparison)

plt.figure(figsize=(12,5))
plt.plot(cum_hold.index, cum_hold.values, label="buy & hold (next-day returns)")
plt.plot(cum_strat.index, cum_strat.values, label="RF sign strategy (after costs)")
plt.legend()
plt.title(f"Simple strategy comparison ({TICKER})")
plt.show()

def summary_returns(series):
    total_ret = series.iloc[-1] - 1.0
    ann = (series.iloc[-1]) ** (252 / len(series)) - 1 if len(series)>0 else np.nan
    return total_ret, ann

print("\nBacktest summary (note: simplified, no slippage, daily returns):")
print(f"Test days: {len(y_test)}")
print(f"Mean strategy daily return (after costs): {strat_ret_after_costs.mean():.6f}")
print(f"Strategy day-win rate (positive daily return): {(strat_ret_after_costs > 0).mean():.3f}")
print(f"Direction accuracy on test: {direction_accuracy(y_test, rf_preds):.3f}")
