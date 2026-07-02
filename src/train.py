from pathlib import Path
import os
import socket

# Keeps training fast and stable on normal laptops.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

FULL_DATA_PATH = DATA_DIR / "house_prices.csv"
SAMPLE_DATA_PATH = DATA_DIR / "house_prices_sample.csv"
MODEL_PATH = MODELS_DIR / "house_price_model.pkl"


def make_folders():
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)


def load_data():
    """Load local data. If full data is missing, try to download it from OpenML."""
    if os.getenv("USE_SAMPLE") == "1":
        print(f"Using sample dataset from: {SAMPLE_DATA_PATH}")
        return pd.read_csv(SAMPLE_DATA_PATH)

    if FULL_DATA_PATH.exists():
        print(f"Loading full dataset from: {FULL_DATA_PATH}")
        return pd.read_csv(FULL_DATA_PATH)

    try:
        socket.setdefaulttimeout(15)
        print("Full dataset not found. Downloading Ames House Prices data from OpenML...")
        ames = fetch_openml(name="house_prices", as_frame=True, parser="auto")
        df = ames.frame
        df.to_csv(FULL_DATA_PATH, index=False)
        print(f"Downloaded and saved full dataset to: {FULL_DATA_PATH}")
        return df
    except Exception as error:
        print("Could not download full data, so using the included sample file.")
        print(f"Reason: {error}")
        return pd.read_csv(SAMPLE_DATA_PATH)


def build_model(X_train):
    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X_train.select_dtypes(include=["object", "category", "string"]).columns

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", encoder)
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=300,
        random_state=42
    )

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])


def main():
    make_folders()

    df = load_data()
    target = "SalePrice"

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' was not found in the data.")

    X = df.drop(columns=[target, "Id"], errors="ignore")
    y = df[target].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_model(X_train)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    predictions = np.maximum(predictions, 0)

    rmse = mean_squared_error(y_test, predictions) ** 0.5
    rmsle = mean_squared_log_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    metrics = pd.DataFrame({
        "metric": ["RMSE", "RMSLE", "R2 Score"],
        "value": [rmse, rmsle, r2]
    })
    metrics.to_csv(OUTPUTS_DIR / "metrics.csv", index=False)

    prediction_table = pd.DataFrame({
        "ActualPrice": y_test.values,
        "PredictedPrice": predictions
    })
    prediction_table.to_csv(OUTPUTS_DIR / "predictions.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, predictions, alpha=0.7)
    plt.xlabel("Actual Sale Price")
    plt.ylabel("Predicted Sale Price")
    plt.title("Actual vs Predicted House Prices")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()

    joblib.dump(model, MODEL_PATH)

    print("Training complete.")
    print(metrics)
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Outputs saved to: {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
