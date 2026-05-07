import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_PATH = "Titanic-Dataset.csv"
TARGET_COL = "Survived"
ARTIFACT_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.pkl")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.pkl")
METADATA_PATH = os.path.join(ARTIFACT_DIR, "metadata.pkl")


def main():
    df = pd.read_csv(DATA_PATH)
    if TARGET_COL not in df.columns:
        raise ValueError(
            "Expected a cleaned Titanic dataset with Survived column in Titanic-Dataset.csv."
        )

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scale_columns = ["Age", "Fare"]
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_train_scaled[scale_columns] = scaler.fit_transform(X_train[scale_columns])

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Keep bin boundaries in metadata so app preprocessing is consistent.
    fare_q1, fare_q2, fare_q3 = X_train["Fare"].quantile([0.25, 0.5, 0.75]).tolist()
    metadata = {
        "feature_columns": X.columns.tolist(),
        "scale_columns": scale_columns,
        "age_bins": [12, 18, 35, 60],
        "fare_bins": [float(fare_q1), float(fare_q2), float(fare_q3)],
        "model_name": "Logistic Regression",
    }

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(metadata, METADATA_PATH)

    print("Model artifacts exported successfully:")
    print(f"- {MODEL_PATH}")
    print(f"- {SCALER_PATH}")
    print(f"- {METADATA_PATH}")


if __name__ == "__main__":
    main()
