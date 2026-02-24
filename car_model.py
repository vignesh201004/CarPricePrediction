import pandas as pd
import datetime
import joblib
import json
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# 1. Load Real Data
df = pd.read_csv('car_data.csv')

# 2. AUTOMATIC COLUMN FIXER
df.columns = [c.replace('_', ' ').title().replace(' ', '_') for c in df.columns]
df.rename(columns={'Km_Driven': 'Kms_Driven', 'Fuel': 'Fuel_Type'}, inplace=True)

# 3. Feature Engineering
curr_year = datetime.datetime.now().year
df['Age'] = curr_year - df['Year']
df_ml = df.drop(['Year', 'Car_Name'], axis=1, errors='ignore')
df_ml = pd.get_dummies(df_ml, drop_first=True)

X = df_ml.drop('Selling_Price', axis=1, errors='ignore')
y = df_ml['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Model
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train)

# 5. Calculate Metrics
predictions = model.predict(X_test)
accuracy = r2_score(y_test, predictions) * 100
error = mean_absolute_error(y_test, predictions)

# 6. Save Assets
joblib.dump(model, 'car_price_model.pkl')
joblib.dump(X.columns.tolist(), 'model_columns.pkl')

with open('model_meta.json', 'w') as f:
    json.dump({'accuracy': round(accuracy, 2), 'mae': round(error, 2)}, f)

print(f"Success! Model trained. Accuracy: {accuracy:.2f}%")