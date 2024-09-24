# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 23:26:22 2024

@author: CZhong
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns

#%%% Preprocessing

# Load and preprocess the data
df = pd.read_csv('merged_dataset.csv')
df['name'] = df['name'].astype('category')
df['timestamps'] = pd.to_datetime(df['timestamps'], unit='D')

# Separate the data
train_df = df.dropna(subset=['food_security_value'])
pred_df = df[df['food_security_value'].isnull()]

# Prepare features and target
features = ['economy_index', 'avg_rainfall_mm', 'conflict_index']
X = train_df[['name', 'timestamps'] + features]
y = train_df['food_security_value']

# Encode categorical variables and process timestamps
X = pd.get_dummies(X, columns=['name'], drop_first=True)
X['days_since_start'] = (X['timestamps'] - X['timestamps'].min()).dt.days
X = X.drop('timestamps', axis=1)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)

#%%%

# Initialize models
models = {
    'OLS (Ridge)': make_pipeline(StandardScaler(), PolynomialFeatures(degree=2), Ridge(alpha=1.0)),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Perform time series cross-validation for each model
results = {name: [] for name in models.keys()}

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name].append((mse, r2))

# Print cross-validation results
for name, scores in results.items():
    print(f"\n{name} - Cross-validation results:")
    for i, (mse, r2) in enumerate(scores):
        print(f"Fold {i+1} - MSE: {mse:.4f}, R2: {r2:.4f}")
    avg_mse = np.mean([score[0] for score in scores])
    avg_r2 = np.mean([score[1] for score in scores])
    print(f"Average - MSE: {avg_mse:.4f}, R2: {avg_r2:.4f}")

# Train final models on all data
for name, model in models.items():
    model.fit(X, y)

# Prepare the prediction set
X_pred = pred_df[['name', 'timestamps'] + features]
X_pred = pd.get_dummies(X_pred, columns=['name'], drop_first=True)
X_pred['days_since_start'] = (X_pred['timestamps'] - train_df['timestamps'].min()).dt.days
X_pred = X_pred.drop('timestamps', axis=1)

for col in X.columns:
    if col not in X_pred.columns:
        X_pred[col] = 0
X_pred = X_pred[X.columns]

# Make predictions with each model
for name, model in models.items():
    pred_df[f'predicted_food_security_value_{name}'] = model.predict(X_pred)

# Print predictions
print("\nPredictions for countries with missing food_security_value:")
print(pred_df[['name', 'timestamps'] + [f'predicted_food_security_value_{name}' for name in models.keys()]])

# Plot feature importances for Random Forest and Gradient Boosting
for name in ['Random Forest', 'Gradient Boosting']:
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': models[name].feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    feature_importance.plot(x='feature', y='importance', kind='bar')
    plt.title(f'Feature Importance in {name} Model')
    plt.tight_layout()
    plt.savefig(f'feature_importance_{name}.png')
    plt.close()

# Plot OLS coefficients
ols_model = models['OLS (Ridge)'].named_steps['ridge']
poly_features = models['OLS (Ridge)'].named_steps['polynomialfeatures']
feature_names = poly_features.get_feature_names_out(X.columns)
ols_coefficients = pd.DataFrame({
    'feature': feature_names,
    'coefficient': ols_model.coef_
}).sort_values('coefficient', key=abs, ascending=False)

plt.figure(figsize=(12, 8))
ols_coefficients.head(20).plot(x='feature', y='coefficient', kind='bar')
plt.title('Top 20 OLS (Ridge) Coefficients')
plt.tight_layout()
plt.savefig('ols_coefficients.png')
plt.close()

#%%% Reconstruct time series

def reconstruct_time_series(original_df, pred_df, model_name):
    reconstructed_df = original_df.copy()
    
    pred_column = [col for col in pred_df.columns if f'predicted_food_security_value_{model_name}' in col]
    if not pred_column:
        print(f"Warning: No prediction column found for {model_name}")
        return reconstructed_df
    pred_column = pred_column[0]
    
    pred_columns = ['name', 'timestamps', pred_column]
    reconstructed_df = pd.merge(reconstructed_df, pred_df[pred_columns], 
                                on=['name', 'timestamps'], how='left')
    
    mask = reconstructed_df['food_security_value'].isnull()
    reconstructed_df.loc[mask, 'food_security_value'] = reconstructed_df.loc[mask, pred_column]
    
    reconstructed_df = reconstructed_df.drop(columns=[pred_column])
    
    return reconstructed_df

# Reconstruct time series for each model
reconstructed_series = {}
for model_name in models.keys():
    try:
        reconstructed_series[model_name] = reconstruct_time_series(df, pred_df, model_name)
        print(f"Successfully reconstructed series for {model_name}")
    except Exception as e:
        print(f"Error reconstructing series for {model_name}: {str(e)}")

# Plots and summary statistics
if reconstructed_series:
    # Plot reconstructed time series for each model
    plt.figure(figsize=(15, 10))
    for model_name, recon_df in reconstructed_series.items():
        sns.lineplot(x='timestamps', y='food_security_value', data=recon_df, label=model_name, ci=None)

    plt.title('Reconstructed Time Series of Food Security Values')
    plt.xticks([])
    plt.xlabel('Timestamp')
    plt.ylabel('Food Security Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig('reconstructed_time_series_all.png')
    plt.close()

    # Plot for each country separately
    # countries = df['name'].unique()
    # for country_name in countries:
    #     plt.figure(figsize=(15, 10))
    #     for model_name, recon_df in reconstructed_series.items():
    #         country_df = recon_df[recon_df['name'] == country_name]
    #         sns.lineplot(x='timestamps', y='food_security_value', data=country_df, label=model_name, ci=None)

    #     plt.title(f'Reconstructed Time Series of Food Security Values for {country_name}')
    #     plt.xlabel('Timestamp')
    #     plt.ylabel('Food Security Value')
    #     plt.legend()
    #     plt.xticks([])
    #     plt.tight_layout()
    #     plt.savefig(f'reconstructed_time_series_{country_name}.png')
    #     plt.close()

    # Summary statistics
    for model_name, recon_df in reconstructed_series.items():
        print(f"\nSummary statistics for {model_name}:")
        print(recon_df.groupby('name')['food_security_value'].describe())

    # Export
    for model_name, recon_df in reconstructed_series.items():
        recon_df.to_csv(f'reconstructed_series_{model_name}.csv', index=False)
        print(f"Reconstructed series for {model_name} saved to 'reconstructed_series_{model_name}.csv'")
else:
    print("No reconstructed series available for plotting.")

print("Completed.")

