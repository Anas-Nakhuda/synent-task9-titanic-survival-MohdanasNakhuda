import os
import joblib
import pandas as pd
import streamlit as st


MODEL_DIR = "artifacts"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.pkl")
def ensure_artifacts():
    """Load pre-trained model artifacts required for prediction."""
    required_files = [MODEL_PATH, SCALER_PATH, METADATA_PATH]
    missing = [p for p in required_files if not os.path.exists(p)]
    if missing:
        missing_names = ", ".join(os.path.basename(p) for p in missing)
        raise FileNotFoundError(
            f"Missing artifacts: {missing_names}. Please run model building export first."
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    metadata = joblib.load(METADATA_PATH)
    return model, scaler, metadata


def safe_qcut_bin(value, bins):
    """Return bin label index using learned static boundaries."""
    if value <= bins[0]:
        return 0
    if value <= bins[1]:
        return 1
    if value <= bins[2]:
        return 2
    return 3


def build_feature_row(
    pclass,
    sex,
    age,
    sibsp,
    parch,
    fare,
    embarked,
    has_cabin,
    title,
    feature_columns,
    age_bins,
    fare_bins,
):
    """Map simple UI inputs to the engineered features used by the model."""
    row = {col: 0 for col in feature_columns}

    row["Pclass"] = pclass
    row["Sex"] = 1 if sex == "female" else 0
    row["Age"] = float(age)
    row["SibSp"] = sibsp
    row["Parch"] = parch
    row["Fare"] = float(fare)
    row["HasCabin"] = 1 if has_cabin else 0

    family_size = sibsp + parch + 1
    row["FamilySize"] = family_size
    row["IsAlone"] = 1 if family_size == 1 else 0

    if "Embarked_Q" in row:
        row["Embarked_Q"] = 1 if embarked == "Q" else 0
    if "Embarked_S" in row:
        row["Embarked_S"] = 1 if embarked == "S" else 0

    if "Title_Miss" in row:
        row["Title_Miss"] = 1 if title == "Miss" else 0
    if "Title_Mr" in row:
        row["Title_Mr"] = 1 if title == "Mr" else 0
    if "Title_Mrs" in row:
        row["Title_Mrs"] = 1 if title == "Mrs" else 0
    if "Title_Rare" in row:
        row["Title_Rare"] = 1 if title == "Rare" else 0

    # Age bins from training metadata.
    if "AgeGroup_Teen" in row:
        row["AgeGroup_Teen"] = 1 if age_bins[0] < age <= age_bins[1] else 0
    if "AgeGroup_Young Adult" in row:
        row["AgeGroup_Young Adult"] = 1 if age_bins[1] < age <= age_bins[2] else 0
    if "AgeGroup_Adult" in row:
        row["AgeGroup_Adult"] = 1 if age_bins[2] < age <= age_bins[3] else 0
    if "AgeGroup_Senior" in row:
        row["AgeGroup_Senior"] = 1 if age > age_bins[3] else 0

    # Fare bins from training metadata.
    fare_bin = safe_qcut_bin(fare, fare_bins)
    if "FareBin_Medium" in row:
        row["FareBin_Medium"] = 1 if fare_bin == 1 else 0
    if "FareBin_High" in row:
        row["FareBin_High"] = 1 if fare_bin == 2 else 0
    if "FareBin_Very High" in row:
        row["FareBin_Very High"] = 1 if fare_bin == 3 else 0

    return pd.DataFrame([row], columns=feature_columns)


def main():
    st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")
    st.title("Titanic Survival Predictor")
    st.caption("Simple UI Predictor for Titanic Survival")

    model, scaler, metadata = ensure_artifacts()
    feature_columns = metadata["feature_columns"]
    scale_columns = metadata["scale_columns"]
    age_bins = metadata.get("age_bins", [12, 18, 35, 60])
    fare_bins = metadata.get("fare_bins", [7.91, 14.45, 31.00])

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            pclass = st.selectbox("Passenger Class", [1, 2, 3], index=2)
            sex = st.selectbox("Sex", ["male", "female"])
            age = st.slider("Age", min_value=1, max_value=80, value=25)
            sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", 0, 8, 0)
            parch = st.number_input("Parents/Children Aboard (Parch)", 0, 6, 0)

        with col2:
            fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0)
            embarked = st.selectbox("Embarked", ["C", "Q", "S"], index=2)
            has_cabin = st.checkbox("Cabin Information Available", value=False)
            title = st.selectbox("Title", ["Mr", "Miss", "Mrs", "Rare"], index=0)

        submit = st.form_submit_button("Predict")

    if submit:
        input_df = build_feature_row(
            pclass=pclass,
            sex=sex,
            age=age,
            sibsp=sibsp,
            parch=parch,
            fare=fare,
            embarked=embarked,
            has_cabin=has_cabin,
            title=title,
            feature_columns=feature_columns,
            age_bins=age_bins,
            fare_bins=fare_bins,
        )

        input_df_scaled = input_df.copy()
        input_df_scaled[scale_columns] = scaler.transform(input_df[scale_columns])

        pred = model.predict(input_df_scaled)[0]
        proba = model.predict_proba(input_df_scaled)[0][1]

        st.subheader("Prediction Result")
        if pred == 1:
            st.success(f"Likely to Survive (probability: {proba:.2%})")
        else:
            st.error(f"Likely Not to Survive (survival probability: {proba:.2%})")

        st.progress(float(proba))


if __name__ == "__main__":
    main()
