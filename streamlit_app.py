import streamlit as st
import pandas as pd
import joblib
from io import BytesIO

st.set_page_config(
    page_title="California House Price Prediction",
    page_icon="🏠",
    layout="centered",
)

import os
import joblib

if not os.path.exists("house_model.joblib"):
    raise Exception("Model file not found. Run train.py first.")

model = joblib.load("house_model.joblib")
features = joblib.load("house_features.joblib")

PRICE_SCALE = 100000
LOWER_MARGIN = 32500
UPPER_MARGIN = 35200


def predict_price(data: pd.DataFrame) -> pd.DataFrame:
    data = data[features]
    predictions = model.predict(data)
    prices = predictions * PRICE_SCALE
    results = pd.DataFrame(
        {
            "predicted_price_$": prices.round(2),
            "lower_price_$": (prices - LOWER_MARGIN).round(2),
            "upper_price_$": (prices + UPPER_MARGIN).round(2),
        }
    )
    return results


def build_input_dataframe(input_values: dict) -> pd.DataFrame:
    return pd.DataFrame([input_values])


def main():
    st.title("California House Price Prediction")
    st.write(
        "Use the form below to predict California house prices with the same model and feature order as your FastAPI app."
    )

    st.sidebar.header("Single prediction")
    med_inc = st.sidebar.number_input("Median income of households (MedInc)", min_value=0.01, value=3.0, step=0.1)
    house_age = st.sidebar.number_input("Median house age (HouseAge)", min_value=0.0, value=20.0, step=1.0)
    ave_rooms = st.sidebar.number_input("Average number of rooms (AveRooms)", min_value=0.1, value=6.0, step=0.1)
    ave_bedrms = st.sidebar.number_input("Average number of bedrooms (AveBedrms)", min_value=0.1, value=1.0, step=0.1)
    population = st.sidebar.number_input("Population in block group (Population)", min_value=0.0, value=1000.0, step=1.0)
    ave_occup = st.sidebar.number_input("Average occupancy (AveOccup)", min_value=0.1, value=3.0, step=0.1)
    latitude = st.sidebar.number_input("Latitude", min_value=-90.0, max_value=90.0, value=34.0, step=0.01)
    longitude = st.sidebar.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-118.0, step=0.01)

    input_values = {
        "MedInc": med_inc,
        "HouseAge": house_age,
        "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms,
        "Population": population,
        "AveOccup": ave_occup,
        "Latitude": latitude,
        "Longitude": longitude,
    }

    if st.sidebar.button("Predict single house price"):
        input_df = build_input_dataframe(input_values)
        prediction = predict_price(input_df).iloc[0]

        st.subheader("Prediction result")
        st.metric("Predicted price ($)", f"{prediction['predicted_price_$']:,}")
        st.write(
            {
                "lower": f"${prediction['lower_price_$']:,}",
                "upper": f"${prediction['upper_price_$']:,}",
            }
        )

    st.markdown("---")
    st.header("Batch prediction from CSV")
    st.write("Upload a CSV file with the required house features to get predictions for every row.")
    st.write(f"Required columns: {', '.join(features)}")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as error:
            st.error(f"Unable to read uploaded file. {error}")
            return

        missing_cols = set(features) - set(df.columns)
        if missing_cols:
            st.error(f"Missing columns: {sorted(missing_cols)}")
            return

        prediction_df = predict_price(df)
        result_df = pd.concat([df.reset_index(drop=True), prediction_df], axis=1)

        st.success("Predictions generated successfully")
        st.dataframe(result_df.head())

        csv_buffer = BytesIO()
        result_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        st.download_button(
            label="Download results as CSV",
            data=csv_buffer,
            file_name="predicted_results.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
