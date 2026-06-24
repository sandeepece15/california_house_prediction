from fastapi import FastAPI,HTTPException,UploadFile,File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from io import BytesIO

app = FastAPI(title="California House Price Prediction API")

import os
import joblib

if not os.path.exists("house_model.joblib"):
    raise Exception("Model file not found. Run train.py first.")

model = joblib.load("house_model.joblib")
features = joblib.load("house_features.joblib")

class HouseFeature(BaseModel):
    MedInc: float = Field(gt=0, description="Median income of households")
    HouseAge: float = Field(ge=0, description="Median house age")
    AveRooms: float = Field(gt=0, description="Average number of rooms")
    AveBedrms: float = Field(gt=0, description="Average number of bedrooms")
    Population: float = Field(ge=0, description="Population in block group")
    AveOccup: float = Field(gt=0, description="Average household occupancy")
    Latitude: float = Field(ge=-90, le=90, description="Latitude")
    Longitude: float = Field(ge=-180, le=180, description="Longitude")


@app.get("/")
def home():
    return {"message": "House Price Prediction API is running"}


@app.post("/predict")
def predict(data: HouseFeature):
    input_df = pd.DataFrame([data.model_dump()])
    input_df = input_df[features]

    prediction = model.predict(input_df)[0]

    predicted_price = prediction * 100000

    return {
        "predicted_price_$": round(predicted_price, 2),
        "price_range_$": {
            "lower": round(predicted_price - 32500, 2),
            "upper": round(predicted_price + 35200, 2)
        }
    }

@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):

    # Validate file type
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are allowed"
        )

    try:
        # Read uploaded CSV
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))

        # Check required columns
        missing_cols = set(features) - set(df.columns)
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns: {list(missing_cols)}"
            )

        # Prepare input
        X = df[features]

        # Predict
        predictions = model.predict(X)

        # Add predictions column
        df["predicted_price_$"] = predictions * 100000

        # Convert DataFrame to CSV in memory
        output = BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)

        # Return file as response
        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=predicted_results.csv"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))