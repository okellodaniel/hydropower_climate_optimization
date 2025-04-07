# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from prophet import Prophet
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

# Load data
energy_df = pd.read_csv('sample_dataset.csv', parse_dates=['date_time'])
climate_df = pd.read_csv('sample_climate.csv', parse_dates=['Date Time'])

print(f"Energy data shape: {energy_df.shape}")
print(f"Climate data shape: {climate_df.shape}")

# Energy Data Preprocessing
def preprocess_energy_data(df):
    """Process the energy consumption data"""
    # Create a copy of the dataframe
    processed_df = df.copy()
    
    # Identify phase type
    processed_df['phase_type'] = np.where(
        (processed_df['v_blue'].notna() & processed_df['v_blue'] != 0.0) | 
        (processed_df['v_yellow'].notna() & processed_df['v_yellow'] != 0.0), 
        'three-phase', 'single-phase'
    )
    
    # Convert kWh to numeric
    processed_df['kwh'] = pd.to_numeric(processed_df['kwh'], errors='coerce')
    
    # Extract user ID from Source column
    processed_df['user_id'] = processed_df['Source'].str.extract(r'consumer_device_\d+_data_user_(\d+)')
    
    # Extract device ID
    processed_df['device_id'] = processed_df['Source'].str.extract(r'consumer_device_(\d+)_data')
    
    # Convert IDs to integers
    processed_df['user_id'] = pd.to_numeric(processed_df['user_id'])
    processed_df['device_id'] = pd.to_numeric(processed_df['device_id'])
    
    # Add normalized power and voltage features
    # For three-phase systems, average the voltages
    processed_df['voltage_avg'] = processed_df.apply(
        lambda row: np.nanmean([
            row['v_red'] if pd.notna(row['v_red']) else 0,
            row['v_blue'] if pd.notna(row['v_blue']) else 0,
            row['v_yellow'] if pd.notna(row['v_yellow']) else 0
        ]), axis=1
    )
    
    # Calculate apparent power (VA)
    processed_df['apparent_power'] = processed_df['voltage_avg'] * processed_df['current']
    
    # Calculate real power (W) using power factor
    processed_df['real_power'] = processed_df['apparent_power'] * processed_df['power_factor']
    
    # Aggregate to daily consumption per user
    daily_energy = processed_df.groupby(['user_id', pd.Grouper(key='date_time', freq='D')]).agg({
        'kwh': 'sum',
        'current': 'mean',
        'power_factor': 'mean',
        'voltage_avg': 'mean',
        'apparent_power': 'mean',
        'real_power': 'mean',
        'phase_type': 'first',
        'device_id': 'first'
    }).reset_index()
    
    return daily_energy, processed_df

daily_energy, processed_energy_df = preprocess_energy_data(energy_df)

# Climate Data Preprocessing
def preprocess_climate_data(df):
    """Process the climate data"""
    # Rename column for consistency
    processed_df = df.rename(columns={'Date Time': 'date_time'})
    
    # Resample to daily data
    climate_daily = processed_df.resample('D', on='date_time').agg({
        'Temperature (°C)': 'mean',
        'Dewpoint Temperature (°C)': 'mean',
        'U Wind Component (m/s)': 'mean',
        'V Wind Component (m/s)': 'mean',
        'Total Precipitation (mm)': 'sum',
        'Snowfall (mm)': 'sum',
        'Snow Cover (%)': 'mean'
    }).reset_index()
    
    # Calculate wind speed
    climate_daily['wind_speed'] = np.sqrt(
        climate_daily['U Wind Component (m/s)']**2 + 
        climate_daily['V Wind Component (m/s)']**2
    )
    
    # Calculate humidex (feels-like temperature)
    # This is a simplification of the humidex formula
    e = 6.11 * np.exp(5417.7530 * ((1/273.16) - (1/(climate_daily['Dewpoint Temperature (°C)'] + 273.16))))
    h = (0.5555) * (e - 10.0)
    climate_daily['humidex'] = climate_daily['Temperature (°C)'] + h
    
    # Calculate heating degree days (HDD) and cooling degree days (CDD)
    base_temp = 18.0  # Standard base temperature
    climate_daily['HDD'] = np.maximum(0, base_temp - climate_daily['Temperature (°C)'])
    climate_daily['CDD'] = np.maximum(0, climate_daily['Temperature (°C)'] - base_temp)
    
    return climate_daily

climate_daily = preprocess_climate_data(climate_df)

# Data Visualization - Exploratory Analysis
def exploratory_analysis(energy_df, climate_df):
    """Perform exploratory analysis on the data"""
    # User consumption distribution
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='user_id', y='kwh', data=energy_df)
    plt.title('Energy Consumption Distribution by User')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('user_consumption_distribution.png')
    plt.close()
    
    # Time series of consumption
    plt.figure(figsize=(14, 7))
    for user_id in energy_df['user_id'].unique():
        user_data = energy_df[energy_df['user_id'] == user_id]
        plt.plot(user_data['date_time'], user_data['kwh'], label=f'User {user_id}')
    
    plt.title('Energy Consumption Time Series by User')
    plt.xlabel('Date')
    plt.ylabel('kWh')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('consumption_time_series.png')
    plt.close()
    
    # Temperature vs Energy scatter plot
    merged_sample = pd.merge(
        energy_df, 
        climate_df[['date_time', 'Temperature (°C)']], 
        on='date_time', 
        how='inner'
    )
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='Temperature (°C)', 
        y='kwh', 
        hue='user_id', 
        data=merged_sample,
        alpha=0.6
    )
    plt.title('Energy Consumption vs Temperature')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('consumption_vs_temperature.png')
    plt.close()
    
    return "Exploratory analysis completed and saved as PNG files"

# Merge energy and climate data
def merge_energy_climate_data(energy_df, climate_df):
    """Merge energy and climate datasets"""
    # Merge on date
    merged_df = pd.merge(
        energy_df,
        climate_df,
        on='date_time',
        how='left'
    )
    
    # Fill missing climate data using forward fill and backward fill
    climate_columns = [
        'Temperature (°C)', 'Dewpoint Temperature (°C)', 
        'wind_speed', 'Total Precipitation (mm)',
        'Snowfall (mm)', 'Snow Cover (%)', 
        'humidex', 'HDD', 'CDD'
    ]
    
    for col in climate_columns:
        if col in merged_df.columns:
            # First forward fill
            merged_df[col] = merged_df[col].ffill()
            # Then backward fill any remaining NAs
            merged_df[col] = merged_df[col].bfill()
            # If still NaN, use the mean
            if merged_df[col].isna().any():
                merged_df[col] = merged_df[col].fillna(merged_df[col].mean())
    
    return merged_df

merged_df = merge_energy_climate_data(daily_energy, climate_daily)

# Feature Engineering
def create_features(df):
    """Create features for time series forecasting"""
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Temporal features
    df_copy['day_of_week'] = df_copy['date_time'].dt.dayofweek
    df_copy['day_of_month'] = df_copy['date_time'].dt.day
    df_copy['month'] = df_copy['date_time'].dt.month
    df_copy['year'] = df_copy['date_time'].dt.year
    df_copy['is_weekend'] = df_copy['day_of_week'].isin([5,6]).astype(int)
    df_copy['quarter'] = df_copy['date_time'].dt.quarter
    
    # Season (meteorological seasons)
    df_copy['season'] = df_copy['month'].apply(lambda x: 
        1 if x in [12, 1, 2] else  # Winter
        2 if x in [3, 4, 5] else   # Spring
        3 if x in [6, 7, 8] else   # Summer
        4                          # Fall
    )
    
    # Is holiday feature (simplified - you'd want a proper holiday calendar)
    # This example uses weekends as a proxy
    df_copy['is_holiday'] = df_copy['is_weekend']
    
    # Weather interaction features
    if 'Temperature (°C)' in df_copy.columns and 'wind_speed' in df_copy.columns:
        # Wind chill effect (simplified)
        df_copy['wind_chill'] = df_copy['Temperature (°C)'] - (
            0.5 * df_copy['wind_speed']
        )
        
        # Create temperature bins
        df_copy['temp_bin'] = pd.cut(
            df_copy['Temperature (°C)'], 
            bins=[-50, 0, 10, 20, 30, 50], 
            labels=[0, 1, 2, 3, 4]
        ).astype('int')
    
    # User-specific time series features
    for user_id in df_copy['user_id'].unique():
        user_mask = df_copy['user_id'] == user_id
        
        # Last 7 days lag
        for lag in [1, 2, 3, 7, 14]:
            df_copy.loc[user_mask, f'kwh_lag_{lag}'] = df_copy.loc[user_mask, 'kwh'].shift(lag)
        
        # Rolling statistics
        for window in [3, 7, 14, 30]:
            # Mean
            rolling = df_copy.loc[user_mask, 'kwh'].rolling(window, min_periods=1).mean()
            df_copy.loc[user_mask, f'kwh_rolling_mean_{window}'] = rolling
            
            # Standard deviation (volatility)
            rolling_std = df_copy.loc[user_mask, 'kwh'].rolling(window, min_periods=1).std()
            df_copy.loc[user_mask, f'kwh_rolling_std_{window}'] = rolling_std.fillna(0)
            
            # Min & Max
            rolling_min = df_copy.loc[user_mask, 'kwh'].rolling(window, min_periods=1).min()
            rolling_max = df_copy.loc[user_mask, 'kwh'].rolling(window, min_periods=1).max()
            df_copy.loc[user_mask, f'kwh_rolling_min_{window}'] = rolling_min
            df_copy.loc[user_mask, f'kwh_rolling_max_{window}'] = rolling_max
    
    # Fill NA values
    numeric_columns = df_copy.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        if df_copy[col].isna().any():
            # Group by user_id to fill NAs by user
            df_copy[col] = df_copy.groupby('user_id')[col].transform(
                lambda x: x.fillna(x.mean() if not np.isnan(x.mean()) else 0)
            )
    
    return df_copy

# Process features
processed_df = create_features(merged_df)

# Check for and handle any remaining NaN values
print(f"NaN values in processed data: {processed_df.isna().sum().sum()}")
if processed_df.isna().sum().sum() > 0:
    processed_df = processed_df.fillna(0)  # Replace any remaining NaNs with zeros

# Feature importance analysis
def analyze_feature_importance(df):
    """Analyze feature importance for energy consumption prediction"""
    # Prepare features
    features = [col for col in df.columns if col not in 
                ['date_time', 'user_id', 'device_id', 'kwh', 'phase_type']]
    
    if len(features) < 2:
        return "Not enough features for importance analysis"
    
    # Train a Random Forest model
    X = df[features]
    y = df['kwh']
    
    # Handle any categorical variables
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if not categorical_cols.empty:
        # One-hot encode categorical features
        ct = ColumnTransformer(
            [('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_cols)],
            remainder='passthrough'
        )
        X = ct.fit_transform(X)
    
    # Train model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    
    # Get feature importances
    if hasattr(rf_model, 'feature_importances_'):
        importances = rf_model.feature_importances_
        
        if len(categorical_cols) > 0:
            # For one-hot encoded features, we need to map back
            # This is a simplification
            feature_names = features.copy()
        else:
            feature_names = features
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances[:len(feature_names)]
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
        plt.title('Feature Importance for Energy Consumption')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        
        return importance_df
    else:
        return "Could not calculate feature importance"

# Analyze feature importance
feature_importance = analyze_feature_importance(processed_df)
print("Top important features:")
print(feature_importance.head(10) if isinstance(feature_importance, pd.DataFrame) else feature_importance)

# Train/Test Split - use last month as test
def train_test_split(df, test_days=30):
    """Split data into training and testing sets"""
    # Get max date for each user
    user_max_dates = df.groupby('user_id')['date_time'].max()
    
    # Initialize empty DataFrames
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    
    # Split by user to ensure each has proper test data
    for user_id, max_date in user_max_dates.items():
        user_data = df[df['user_id'] == user_id]
        cutoff_date = max_date - pd.Timedelta(days=test_days)
        
        user_train = user_data[user_data['date_time'] <= cutoff_date]
        user_test = user_data[user_data['date_time'] > cutoff_date]
        
        train_df = pd.concat([train_df, user_train])
        test_df = pd.concat([test_df, user_test])
    
    return train_df, test_df

# Split data
train_df, test_df = train_test_split(processed_df)
print(f"Training data: {train_df.shape}")
print(f"Testing data: {test_df.shape}")

# Prophet Model with Hyperparameter Optimization
def optimize_prophet(user_df):
    """Find optimal hyperparameters for Prophet model"""
    best_mae = float('inf')
    best_params = {}
    
    # Define hyperparameter grid
    changepoint_prior_scales = [0.001, 0.01, 0.1, 0.5]
    seasonality_prior_scales = [0.01, 0.1, 1.0, 10.0]
    seasonality_modes = ['additive', 'multiplicative']
    
    # Split data for validation
    cutoff_idx = int(len(user_df) * 0.8)
    train_data = user_df.iloc[:cutoff_idx]
    val_data = user_df.iloc[cutoff_idx:]
    
    # Prepare data for Prophet
    prophet_train = train_data.rename(columns={'date_time': 'ds', 'kwh': 'y'})
    prophet_val = val_data.rename(columns={'date_time': 'ds', 'kwh': 'y'})
    
    # Grid search
    for cp_scale in changepoint_prior_scales:
        for s_scale in seasonality_prior_scales:
            for s_mode in seasonality_modes:
                try:
                    model = Prophet(
                        changepoint_prior_scale=cp_scale,
                        seasonality_prior_scale=s_scale,
                        seasonality_mode=s_mode,
                        daily_seasonality=False,
                        weekly_seasonality=True,
                        yearly_seasonality=True
                    )
                    
                    # Add regressors if available
                    for feature in ['Temperature (°C)', 'wind_speed', 'HDD', 'CDD']:
                        if feature in user_df.columns:
                            model.add_regressor(feature)
                    
                    # Fit the model
                    model.fit(prophet_train)
                    
                    # Predict on validation
                    future = pd.DataFrame({'ds': prophet_val['ds']})
                    for feature in ['Temperature (°C)', 'wind_speed', 'HDD', 'CDD']:
                        if feature in user_df.columns:
                            future[feature] = prophet_val[feature].values
                    
                    forecast = model.predict(future)
                    
                    # Calculate error
                    mae = mean_absolute_error(prophet_val['y'], forecast['yhat'])
                    
                    if mae < best_mae:
                        best_mae = mae
                        best_params = {
                            'changepoint_prior_scale': cp_scale,
                            'seasonality_prior_scale': s_scale,
                            'seasonality_mode': s_mode
                        }
                except Exception as e:
                    print(f"Error with params {cp_scale}, {s_scale}, {s_mode}: {e}")
                    continue
    
    return best_params

# Prophet Model Implementation
def run_prophet_forecast(user_df, periods=30, optimize=True):
    """Run Prophet forecasting model for a single user"""
    if len(user_df) < 14:
        # Not enough data for Prophet
        print(f"Not enough data for Prophet model ({len(user_df)} points)")
        last_date = user_df['date_time'].max()
        mean_kwh = user_df['kwh'].mean() if len(user_df) > 0 else 0
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)
        forecast_df = pd.DataFrame({
            'ds': future_dates,
            'yhat': [mean_kwh] * periods
        })
        return forecast_df
    
    try:
        # Get optimal parameters if requested
        params = optimize_prophet(user_df) if optimize else {}
        
        # Default parameters if optimization failed
        if not params:
            params = {
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10.0,
                'seasonality_mode': 'multiplicative'
            }
        
        # Create the model with optimized parameters
        model = Prophet(
            changepoint_prior_scale=params['changepoint_prior_scale'],
            seasonality_prior_scale=params['seasonality_prior_scale'],
            seasonality_mode=params['seasonality_mode'],
            weekly_seasonality=True,
            yearly_seasonality=True,
            daily_seasonality=False
        )
        
        # Add country holidays if available
        try:
            model.add_country_holidays(country_name='US')
        except:
            # If holidays package is not available
            pass
        
        # Add regressors if available
        for feature in ['Temperature (°C)', 'wind_speed', 'HDD', 'CDD']:
            if feature in user_df.columns and not user_df[feature].isna().all():
                model.add_regressor(feature)
        
        # Prepare dataframe for Prophet
        prophet_df = user_df.rename(columns={
            'date_time': 'ds',
            'kwh': 'y'
        })
        
        # Select columns for Prophet
        prophet_cols = ['ds', 'y'] + [col for col in ['Temperature (°C)', 'wind_speed', 'HDD', 'CDD'] 
                           if col in user_df.columns and not user_df[col].isna().all()]
        prophet_df = prophet_df[prophet_cols]
        
        # Handle zero values (add small value to prevent issues with multiplicative seasonality)
        if params['seasonality_mode'] == 'multiplicative':
            prophet_df['y'] = prophet_df['y'].replace(0, 1e-6)
        
        # Fit the model
        model.fit(prophet_df)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods)
        
        # Add regressor values to future dataframe
        # Use a more realistic approach here if you have access to weather forecasts
        for feature in ['Temperature (°C)', 'wind_speed', 'HDD', 'CDD']:
            if feature in prophet_df.columns:
                # If it's a seasonal feature, use seasonal pattern
                if feature in ['Temperature (°C)', 'HDD', 'CDD']:
                    # Get data from same month last year if available
                    last_date = prophet_df['ds'].max()
                    forecast_start = last_date + pd.Timedelta(days=1)
                    forecast_dates = pd.date_range(start=forecast_start, periods=periods)
                    
                    same_month_last_year = prophet_df[
                        (prophet_df['ds'].dt.month.isin(forecast_dates.month)) & 
                        (prophet_df['ds'].dt.day.isin(forecast_dates.day))
                    ]
                    
                    if len(same_month_last_year) > 0:
                        # Use values from same period last year
                        future[feature] = np.nan
                        for i, date in enumerate(forecast_dates):
                            same_day_last_year = same_month_last_year[
                                (same_month_last_year['ds'].dt.month == date.month) & 
                                (same_month_last_year['ds'].dt.day == date.day)
                            ]
                            if len(same_day_last_year) > 0:
                                future.loc[future['ds'] == date, feature] = same_day_last_year[feature].values[0]
                            else:
                                # Fallback to average
                                future.loc[future['ds'] == date, feature] = prophet_df[feature].mean()
                    else:
                        # Use monthly averages
                        for i, date in enumerate(forecast_dates):
                            month_avg = prophet_df[prophet_df['ds'].dt.month == date.month][feature].mean()
                            if np.isnan(month_avg):
                                month_avg = prophet_df[feature].mean()
                            future.loc[future['ds'] == date, feature] = month_avg
                else:
                    # For non-seasonal features, use mean
                    future[feature] = prophet_df[feature].mean()
        
        # Make forecast
        forecast = model.predict(future)
        
        # Ensure all forecast dates are included
        forecast_dates = pd.date_range(start=user_df['date_time'].max() + pd.Timedelta(days=1), periods=periods)
        if len(forecast) < len(forecast_dates) + len(prophet_df):
            # Fill in missing forecast dates with mean
            missing_dates = pd.DataFrame({
                'ds': [d for d in forecast_dates if d not in forecast['ds'].values],
                'yhat': [prophet_df['y'].mean()] * len([d for d in forecast_dates if d not in forecast['ds'].values])
            })
            forecast = pd.concat([forecast, missing_dates])
        
        return forecast[['ds', 'yhat']]
    
    except Exception as e:
        print(f"Prophet forecasting failed: {e}")
        # Fall back to a simple forecast
        last_date = user_df['date_time'].max()
        mean_kwh = user_df['kwh'].mean() if len(user_df) > 0 else 0
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)
        forecast_df = pd.DataFrame({
            'ds': future_dates,
            'yhat': [mean_kwh] * periods
        })
        return forecast_df

# ARIMA/SARIMA Model
def run_sarima_forecast(user_df, periods=30):
    """Run SARIMA forecasting model for a single user"""
    if len(user_df) < 30:
        # Not enough data for SARIMA
        print(f"Not enough data for SARIMA model ({len(user_df)} points)")
        return None
    
    try:
        # Prepare data
        ts_data = user_df.set_index('date_time')['kwh']
        
        # Check if we have enough data for seasonal component
        if len(ts_data) >= 14:  # At least 2 weeks for weekly seasonality
            # Fit SARIMA model
            # Order (p,d,q) and seasonal order (P,D,Q,s)
            # This is a starting point - ideally you would perform order selection
            model = SARIMAX(
                ts_data,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 7),  # Weekly seasonality
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            results = model.fit(disp=False)
            
            # Generate forecast
            forecast = results.get_forecast(steps=periods)
            forecast_values = forecast.predicted_mean
            
            # Create forecast dataframe
            last_date = user_df['date_time'].max()
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)
            
            forecast_df = pd.DataFrame({
                'ds': forecast_dates,
                'yhat': forecast_values.values
            })
            
            return forecast_df
        else:
            return None
    
    except Exception as e:
        print(f"SARIMA forecasting failed: {e}")
        return None

# XGBoost Model
def train_xgb_model(train_df, test_df=None):
    """Train an XGBoost model for energy forecasting"""
    # Define features
    features = [col for col in train_df.columns if col not in 
               ['date_time', 'user_id', 'device_id', 'kwh', 'phase_type']]
    
    # Check if we have enough features
    if len(features) < 3:
        print("Not enough features for XGBoost model")
        return None, None, None
    
    X_train = train_df[features]
    y_train = train_df['kwh']
    
    # Handle categorical features
    cat_features = X_train.select_dtypes(include=['object', 'category']).columns
    
    # Create preprocessing pipeline
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features) if len(cat_features) > 0 else ('pass', 'passthrough', [])
        ],
        remainder='passthrough'
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ))
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Evaluate if test data is available
    if test_df is not None and len(test_df) > 0:
        X_test = test_df[features]
        y_test = test_df['kwh']
        
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"XGBoost Model Evaluation:")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
    
    return pipeline, features, preprocessor

# Ensemble forecasting
def generate_ensemble_forecasts(df, forecast_days=30):
    """Generate forecasts using an ensemble of models"""
    forecasts = {}
    unique_users = df['user_id'].unique()
    
    # Generate forecasts for each user
    for user_id in unique_users:
        print(f"Generating forecasts for User {user_id}")
        user_df = df[df['user_id'] == user_id].copy()
        
        # Check if we have enough data
        if len(user_df) < 14:
            print(f"Not enough data for User {user_id} to generate reliable forecasts")
            continue
        
        user_forecasts = []
        
        # 1. Prophet forecast
        try:
            prophet_forecast = run_prophet_forecast(user_df, periods=forecast_days, optimize=True)
            if prophet_forecast is not None:
                prophet_forecast['model'] = 'Prophet'
                prophet_forecast['user_id'] = user_id
                user_forecasts.append(prophet_forecast)
                print(f"  - Prophet forecast generated")
        except Exception as e:
            print(f"  - Prophet forecast failed: {e}")
        
        # 2. SARIMA forecast
        try:
            sarima_forecast = run_sarima_forecast(user_df, periods=forecast_days)
            if sarima_forecast is not None:
                sarima_forecast['model'] = 'SARIMA'
                sarima_forecast['user_id'] = user_id
                user_forecasts.append(sarima_forecast)
                print(f"  - SARIMA forecast generated")
        except Exception as e:
            print(f"  - SARIMA forecast failed: {e}")
        
        # 3. XGBoost forecast (requires future values of features)
        try:
            # Train XGBoost model on available data
            model, features, preprocessor = train_xgb_model(user_df)
            
            if model is not None and len(features) > 0:
                # Generate future dates
                last_date = user_df['date_time'].max()
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
                
                # Create future feature dataframe
                future_df = pd.DataFrame({'date_time': future_dates})
                future_df['user_id'] = user_id
                
                # Add temporal features
                future_df['day_of_week'] = future_df['date_time'].dt.dayofweek
                future_df['day_of_month'] = future_df['date_time'].dt.day
                future_df['month'] = future_df['date_time'].dt.month
                future_df['year'] = future_df['date_time'].dt.year
                future_df['is_weekend'] = future_df['day_of_week'].isin([5,6]).astype(int)
                future_df['quarter'] = future_df['date_time'].dt.quarter
                future_df['season'] = future_df['month'].apply(lambda x: 
                    1 if x in [12, 1, 2] else  # Winter
                    2 if x in [3, 4, 5] else   # Spring
                    3 if x in [6, 7, 8] else   # Summer
                    4                          # Fall
                )
                future_df['is_holiday'] = future_df['is_weekend']  # Simplified
                
                # Add climate features (using averages from historical data)
                for feature in [col for col in features if col in ['Temperature (°C)', 'wind_speed', 'HDD', 'CDD', 
                                                                'Total Precipitation (mm)', 'Snow Cover (%)']]:
                    # Use monthly average
                    for month in range(1, 13):
                        month_mask = future_df['month'] == month
                        historical_avg = user_df[user_df['month'] == month][feature].mean()
                        
                        if np.isnan(historical_avg):
                            historical_avg = user_df[feature].mean()  # Fallback to overall average
                        
                        future_df.loc[month_mask, feature] = historical_avg
                
                # Add lag features (using last known values)
                for lag in [1, 2, 3, 7, 14]:
                    if f'kwh_lag_{lag}' in features:
                        # For the first future date, use actual values
                        last_known_values = user_df['kwh'].iloc[-lag:].tolist() if len(user_df) >= lag else []
                        if len(last_known_values) < lag:
                            last_known_values = [user_df['kwh'].mean()] * (lag - len(last_known_values)) + last_known_values
                        
                        future_df[f'kwh_lag_{lag}'] = np.nan
                        
                        # Fill the first rows with known values
                        for i in range(min(lag, forecast_days)):
                            if i < len(last_known_values):
                                future_df.iloc[i, future_df.columns.get_loc(f'kwh_lag_{lag}')] = last_known_values[-lag+i]
                
                # Add rolling features (initialize with historical values)
                for window in [3, 7, 14, 30]:
                    if f'kwh_rolling_mean_{window}' in features:
                        future_df[f'kwh_rolling_mean_{window}'] = user_df['kwh'].iloc[-window:].mean() if len(user_df) >= window else user_df['kwh'].mean()
                        future_df[f'kwh_rolling_std_{window}'] = user_df['kwh'].iloc[-window:].std() if len(user_df) >= window else 0
                        future_df[f'kwh_rolling_min_{window}'] = user_df['kwh'].iloc[-window:].min() if len(user_df) >= window else user_df['kwh'].mean()
                        future_df[f'kwh_rolling_max_{window}'] = user_df['kwh'].iloc[-window:].max() if len(user_df) >= window else user_df['kwh'].mean()
                
                # Make predictions iteratively for each future date
                xgb_forecasts = []
                
                for i in range(forecast_days):
                    # Prepare features for this date
                    X_future = future_df.iloc[i:i+1]
                    
                    # Ensure all needed features are present
                    for feature in features:
                        if feature not in X_future.columns:
                            # Use mean value from training data
                            if feature in user_df.columns:
                                X_future[feature] = user_df[feature].mean()
                            else:
                                X_future[feature] = 0  # Fallback value
                    
                    # Make prediction
                    y_pred = model.predict(X_future[features])
                    xgb_forecasts.append(float(y_pred[0]))
                    
                    # Update lag features for next prediction
                    if i + 1 < forecast_days:
                        for lag in [1, 2, 3, 7, 14]:
                            if f'kwh_lag_{lag}' in features and i + 1 >= lag:
                                future_df.iloc[i+1, future_df.columns.get_loc(f'kwh_lag_{lag}')] = xgb_forecasts[-lag] if i+1 >= lag else user_df['kwh'].iloc[-lag+i+1]
                
                # Create forecast dataframe
                xgb_forecast_df = pd.DataFrame({
                    'ds': future_dates,
                    'yhat': xgb_forecasts,
                    'model': 'XGBoost',
                    'user_id': user_id
                })
                
                user_forecasts.append(xgb_forecast_df)
                print(f"  - XGBoost forecast generated")
        except Exception as e:
            print(f"  - XGBoost forecast failed: {e}")
        
        # Combine all forecasts for this user
        if user_forecasts:
            combined_forecast = pd.concat(user_forecasts)
            
            # Create ensemble forecast (simple average)
            ensemble_df = combined_forecast.pivot_table(
                index=['ds', 'user_id'],
                columns='model',
                values='yhat'
            ).reset_index()
            
            # Calculate ensemble (average of available forecasts)
            model_cols = [col for col in ensemble_df.columns if col not in ['ds', 'user_id']]
            ensemble_df['Ensemble'] = ensemble_df[model_cols].mean(axis=1)
            
            # Melt back to long format
            ensemble_df = pd.melt(
                ensemble_df,
                id_vars=['ds', 'user_id'],
                value_vars=model_cols + ['Ensemble'],
                var_name='model',
                value_name='yhat'
            )
            
            forecasts[user_id] = ensemble_df
        else:
            print(f"No forecasts generated for User {user_id}")
    
    return forecasts

# Forecast evaluation
def evaluate_forecasts(forecasts, actual_df):
    """Evaluate forecast performance against actual data"""
    results = []
    
    for user_id, forecast_df in forecasts.items():
        user_actual = actual_df[actual_df['user_id'] == user_id]
        
        for model in forecast_df['model'].unique():
            model_forecast = forecast_df[forecast_df['model'] == model]
            
            # Merge forecast with actual data
            evaluation_df = pd.merge(
                model_forecast,
                user_actual[['date_time', 'kwh']],
                left_on='ds',
                right_on='date_time',
                how='inner'
            )
            
            if len(evaluation_df) > 0:
                # Calculate metrics
                mae = mean_absolute_error(evaluation_df['kwh'], evaluation_df['yhat'])
                rmse = np.sqrt(mean_squared_error(evaluation_df['kwh'], evaluation_df['yhat']))
                mape = np.mean(np.abs((evaluation_df['kwh'] - evaluation_df['yhat']) / (evaluation_df['kwh'] + 1e-6))) * 100
                
                results.append({
                    'user_id': user_id,
                    'model': model,
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'n_points': len(evaluation_df)
                })
    
    if results:
        results_df = pd.DataFrame(results)
        return results_df
    else:
        return "No evaluation data available"

# Visualization functions
def plot_forecasts(forecasts, actual_df=None, highlight_ensemble=True):
    """Plot forecasts for each user"""
    for user_id, forecast_df in forecasts.items():
        plt.figure(figsize=(14, 8))
        
        # Filter actual data for this user
        if actual_df is not None:
            user_actual = actual_df[actual_df['user_id'] == user_id]
            plt.plot(user_actual['date_time'], user_actual['kwh'], 
                     'k.-', label='Actual', alpha=0.7)
        
        # Plot each model's forecast
        for model in forecast_df['model'].unique():
            model_data = forecast_df[forecast_df['model'] == model]
            
            if model == 'Ensemble' and highlight_ensemble:
                plt.plot(model_data['ds'], model_data['yhat'], 
                         'r-', linewidth=3, label=model)
            else:
                plt.plot(model_data['ds'], model_data['yhat'], 
                         '--', label=model, alpha=0.7)
        
        plt.title(f'Energy Consumption Forecast for User {user_id}')
        plt.xlabel('Date')
        plt.ylabel('kWh')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'forecast_user_{user_id}.png')
        plt.close()

def generate_user_report(user_id, forecast_df, actual_df=None, evaluation_df=None):
    """Generate a detailed report for a single user"""
    plt.figure(figsize=(16, 12))
    
    # Plot layout
    gs = plt.GridSpec(3, 2)
    
    # 1. Historical and forecast plot
    ax1 = plt.subplot(gs[0, :])
    
    # Plot historical data if available
    if actual_df is not None:
        user_actual = actual_df[actual_df['user_id'] == user_id]
        ax1.plot(user_actual['date_time'], user_actual['kwh'], 
                'k.-', label='Historical', alpha=0.7)
    
    # Plot forecast
    for model in forecast_df['model'].unique():
        model_data = forecast_df[forecast_df['model'] == model]
        if model == 'Ensemble':
            ax1.plot(model_data['ds'], model_data['yhat'], 
                    'r-', linewidth=3, label=model)
        else:
            ax1.plot(model_data['ds'], model_data['yhat'], 
                    '--', label=model, alpha=0.6)
    
    ax1.set_title(f'Energy Consumption Forecast for User {user_id}')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('kWh')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Model performance comparison if evaluation data exists
    if evaluation_df is not None and isinstance(evaluation_df, pd.DataFrame):
        user_eval = evaluation_df[evaluation_df['user_id'] == user_id]
        
        if len(user_eval) > 0:
            ax2 = plt.subplot(gs[1, 0])
            sns.barplot(x='model', y='mae', data=user_eval, ax=ax2)
            ax2.set_title('Mean Absolute Error by Model')
            ax2.set_ylabel('MAE (kWh)')
            ax2.set_xlabel('')
            ax2.tick_params(axis='x', rotation=45)
            
            ax3 = plt.subplot(gs[1, 1])
            sns.barplot(x='model', y='mape', data=user_eval, ax=ax3)
            ax3.set_title('Mean Absolute Percentage Error by Model')
            ax3.set_ylabel('MAPE (%)')
            ax3.set_xlabel('')
            ax3.tick_params(axis='x', rotation=45)
    
    # 3. Monthly consumption pattern
    if actual_df is not None:
        user_actual = actual_df[actual_df['user_id'] == user_id]
        
        ax4 = plt.subplot(gs[2, 0])
        user_actual['month'] = user_actual['date_time'].dt.month
        monthly_consumption = user_actual.groupby('month')['kwh'].mean().reindex(range(1, 13))
        monthly_consumption.plot(kind='bar', ax=ax4)
        ax4.set_title('Average Monthly Consumption')
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Average kWh')
        ax4.set_xticks(range(12))
        ax4.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        
        # 4. Weekday vs weekend consumption
        ax5 = plt.subplot(gs[2, 1])
        user_actual['is_weekend'] = user_actual['date_time'].dt.dayofweek >= 5
        weekend_consumption = user_actual.groupby('is_weekend')['kwh'].mean()
        weekend_consumption.index = ['Weekday', 'Weekend']
        weekend_consumption.plot(kind='bar', ax=ax5)
        ax5.set_title('Weekday vs Weekend Consumption')
        ax5.set_ylabel('Average kWh')
        ax5.set_xlabel('')
    
    plt.tight_layout()
    plt.savefig(f'report_user_{user_id}.png')
    plt.close()
    
    return f'report_user_{user_id}.png'

# Run the full forecasting pipeline
def run_forecasting_pipeline(energy_df, climate_df, forecast_days=30):
    """Run the complete forecasting pipeline"""
    print("1. Preprocessing data...")
    daily_energy, processed_energy_df = preprocess_energy_data(energy_df)
    climate_daily = preprocess_climate_data(climate_df)
    
    print("2. Merging datasets...")
    merged_df = merge_energy_climate_data(daily_energy, climate_daily)
    
    print("3. Feature engineering...")
    processed_df = create_features(merged_df)
    
    print("4. Analyzing feature importance...")
    feature_importance = analyze_feature_importance(processed_df)
    print("Top important features:")
    print(feature_importance.head(10) if isinstance(feature_importance, pd.DataFrame) else feature_importance)
    
    print("5. Splitting data for training and testing...")
    train_df, test_df = train_test_split(processed_df)
    
    print("6. Generating forecasts...")
    forecasts = generate_ensemble_forecasts(train_df, forecast_days=forecast_days)
    
    print("7. Evaluating forecasts...")
    evaluation_df = evaluate_forecasts(forecasts, test_df)
    
    print("8. Generating visualizations...")
    plot_forecasts(forecasts, test_df)
    
    print("9. Generating user reports...")
    for user_id in forecasts.keys():
        generate_user_report(user_id, forecasts[user_id], processed_df, evaluation_df)
    
    print("10. Forecasting completed!")
    
    return forecasts, evaluation_df

# Run the full pipeline
if __name__ == "__main__":
    # Run exploratory analysis
    exploratory_analysis(daily_energy, climate_daily)
    
    # Run the forecasting pipeline
    forecasts, evaluation = run_forecasting_pipeline(energy_df, climate_df, forecast_days=30)
    
    # Print overall evaluation results
    if isinstance(evaluation, pd.DataFrame):
        print("\nOverall Model Performance:")
        print(evaluation.groupby('model')[['mae', 'rmse', 'mape']].mean())
        
        # Find best model overall
        best_model = evaluation.groupby('model')['mae'].mean().idxmin()
        print(f"\nBest performing model overall: {best_model}")
        
        # Save evaluation results
        evaluation.to_csv('forecast_evaluation.csv', index=False)
    
    print("\nForecasting completed! Results saved to disk.")