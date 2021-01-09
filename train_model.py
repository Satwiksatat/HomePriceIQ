import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the dataset
print("Loading data...")
df_train = pd.read_csv('./data/train.csv')

# Create a copy of the dataframe for preprocessing
df = df_train.copy()

# Handle missing values
print("Handling missing values...")
# For numeric columns, fill with median
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# Feature Engineering
print("Creating new features...")
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
df['TotalBathrooms'] = df['FullBath']  # Simplified for the demo
df['HouseAge'] = df['YrSold'] - df['YearBuilt']

# Select only the features we want to use
selected_features = [
    'LotArea',
    'YearBuilt',
    'TotalBsmtSF',
    '1stFlrSF',
    '2ndFlrSF',
    'BedroomAbvGr',
    'FullBath',
    'GarageArea',
    'OverallQual',
    'TotalSF',
    'TotalBathrooms',
    'HouseAge'
]

# Log transform the target variable
df['SalePrice_Log'] = np.log1p(df['SalePrice'])

# Prepare features for modeling
features = df[selected_features]
target = df['SalePrice_Log']

# Split the data
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize and train models
print("Training models...")
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate models
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, features, target, cv=5, scoring='r2')
    
    results[name] = {
        'RMSE': rmse,
        'R2': r2,
        'CV R2 Mean': cv_scores.mean(),
        'CV R2 Std': cv_scores.std()
    }

# Print results
print("\nModel Results:")
for name, metrics in results.items():
    print(f"\n{name}:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

# Save the best model (XGBoost)
print("\nSaving the best model (XGBoost)...")
best_model = models['XGBoost']
joblib.dump(best_model, './models/house_price_model.joblib')
print("Model saved as './models/house_price_model.joblib'") 