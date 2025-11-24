"""
Simple LSTM starter — trains on sequences of features to predict next-day return.
This is a short template; tune train/test split, sequence length, scaling, and early stopping.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from features import compute_features

def create_sequences(X, y, seq_len=20):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len].values)
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="AAPL")
    p.add_argument("--data_csv", default=None)
    p.add_argument("--seq_len", type=int, default=20)
    args = p.parse_args()

    csv_path = Path(args.data_csv) if args.data_csv else Path("data") / f"{args.ticker}.csv"
    df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
    df = compute_features(df)
    df = df.dropna()
    feature_cols = [c for c in df.columns if c.startswith("lag_") or c.startswith("roll_") or c.startswith("sma_") or c in ("rsi_14", "vol_rolling_10")]
    X = df[feature_cols]
    y = df["target_ret_1"]

    # split by time (80% train)
    split_idx = int(len(df) * 0.8)
    X_train_raw, X_test_raw = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # scale with train-only scaler
    scaler = StandardScaler().fit(X_train_raw)
    X_train = pd.DataFrame(scaler.transform(X_train_raw), index=X_train_raw.index, columns=feature_cols)
    X_test = pd.DataFrame(scaler.transform(X_test_raw), index=X_test_raw.index, columns=feature_cols)

    # create sequences
    seq_len = args.seq_len
    X_tr, y_tr = create_sequences(X_train, y_train, seq_len=seq_len)
    X_te, y_te = create_sequences(X_test, y_test, seq_len=seq_len)

    model = Sequential([
        LSTM(64, input_shape=(seq_len, X_tr.shape[2]), return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(X_tr, y_tr, validation_split=0.1, epochs=50, batch_size=32, callbacks=[es])
    preds = model.predict(X_te).ravel()
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = mean_absolute_error(y_te, preds)
    rmse = mean_squared_error(y_te, preds, squared=False)
    print(f"LSTM Test MAE: {mae:.6f}, RMSE: {rmse:.6f}")

if __name__ == "__main__":
    main()
