# Titanic Survival Prediction

## Problem Statement

The goal of this project is to build a machine learning model that predicts whether a passenger survived the Titanic disaster.  
Using passenger information such as age, sex, class, fare, and family-related fields, the model learns survival patterns and provides a survival prediction for new inputs.

## Dataset Details

- **Dataset used:** `Titanic-Dataset.csv`
- **Target column:** `Survived` (0 = did not survive, 1 = survived)
- **Important input features:** `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`, and engineered features from preprocessing
- **Project notebooks and scripts:**
  - `01_eda.ipynb` - exploratory data analysis
  - `02_data_cleaning.ipynb` - data cleaning and feature engineering
  - `03_model_building.ipynb` - model training and evaluation
  - `train_model.py` - exports production-ready artifacts
  - `app.py` - Streamlit app for live prediction

## Approach

1. **Data understanding (EDA):** explored distributions, missing values, and relationships with survival.
2. **Data preprocessing:** handled missing data, encoded categorical columns, and prepared model-ready features.
3. **Feature engineering:** created useful derived features to improve model performance.
4. **Model training and comparison:** trained multiple models and selected the best-performing one.
5. **Artifact export:** saved reusable files in `artifacts/`:
   - `model.pkl`
   - `scaler.pkl`
   - `metadata.pkl`
6. **Prediction interface:** built a Streamlit UI for manual input and real-time survival prediction.

## Results

- Built an end-to-end Titanic survival prediction pipeline from raw data to deployable model artifacts.
- Completed model evaluation and selected the best model for inference.
- Delivered a working Streamlit app (`app.py`) that predicts survival probability from user inputs.
- The project is reproducible and ready for local deployment, with cloud deployment support possible via Streamlit platforms.

## How to Run

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train model and export artifacts**
   ```bash
   python train_model.py
   ```
   This generates:
   - `artifacts/model.pkl`
   - `artifacts/scaler.pkl`
   - `artifacts/metadata.pkl`

5. **Run Streamlit app**
   ```bash
   streamlit run app.py
   ```

6. **Use the app**
   - Open the local URL shown in terminal (usually `http://localhost:8501`)
   - Enter passenger details
   - Click **Predict** to view survival prediction and probability
