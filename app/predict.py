from pathlib import Path

import joblib
import pandas as pd


INPUT_PATH = Path("artifacts/features/weather_features.csv")
MODEL_PATH = Path("artifacts/model.pkl")
OUTPUT_DIR = Path("artifacts/predictions")
OUTPUT_PATH = OUTPUT_DIR / "predictions.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_prediction():
    df = pd.read_csv(INPUT_PATH)
    model = joblib.load(MODEL_PATH)

    X = df.drop(columns=[
        "date",
        "location",
        "theoretical_energy",
        "target_energy_next_hour"
    ])

    predictions = model.predict(X)

    results = df[["date", "location", "target_energy_next_hour"]].copy()
    results["predicted_energy_next_hour"] = predictions

    results.to_csv(OUTPUT_PATH, index=False)

    print(f"Predictions saved to: {OUTPUT_PATH}")
    print(results.tail())


if __name__ == "__main__":
    run_prediction()
