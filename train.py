from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.metrics import mean_absolute_error, r2_score

print("Loading dataset...")

data = fetch_california_housing()

x = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

print("Training model...")

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

err = mean_absolute_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print(f"MAE: {err}")
print(f"R2 Score: {r2}")

print("Saving model files...")

joblib.dump(model, "house_model.joblib")
joblib.dump(list(x.columns), "house_features.joblib")

print("Done!")