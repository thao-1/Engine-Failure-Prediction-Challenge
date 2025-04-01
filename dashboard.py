import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import norm

# Set page title and layout
st.set_page_config(page_title="Engine Health Monitoring", layout="wide")
st.title("Engine Health Monitoring Dashboard")

# Load data function
@st.cache_data
def load_data():
    # Define column names for the dataset (matches your model training)
    columns = ['Engine_ID', 'Cycle', 'Setting_1', 'Setting_2', 'Setting_3'] + [f'Sensor_{i}' for i in range(1, 22)]
    
    # Load training and test datasets
    train_df = pd.read_csv("train_FD001.txt", sep=" ", header=None, names=columns, engine='python')
    test_df = pd.read_csv("test_FD001.txt", sep=" ", header=None, names=columns, engine='python')
    rul_df = pd.read_csv("RUL_FD001.txt", sep=" ", header=None, names=['RUL'])
    
    # Clean dataset (remove empty columns due to extra spaces in txt file)
    train_df = train_df.dropna(axis=1, how='all')
    test_df = test_df.dropna(axis=1, how='all')
    
    # Feature Engineering: Create RUL Target Column
    train_df['RUL'] = train_df.groupby('Engine_ID')['Cycle'].transform(lambda x: max(x) - x)
    
    # Create additional time-based features
    train_df['Cycle_Squared'] = train_df['Cycle'] ** 2
    test_df['Cycle_Squared'] = test_df['Cycle'] ** 2
    
    # Handle anomalous values by clipping extreme sensor readings (1-19 as per your code)
    for col in [f'Sensor_{i}' for i in range(1, 20)]:
        train_df[col] = train_df[col].clip(lower=train_df[col].quantile(0.01), upper=train_df[col].quantile(0.99))
        test_df[col] = test_df[col].clip(lower=test_df[col].quantile(0.01), upper=test_df[col].quantile(0.99))
    
    # Normalize sensor readings (only sensors 1-19 as per your training)
    sensor_columns = [f'Sensor_{i}' for i in range(1, 20)]
    scaler = StandardScaler()
    train_df[sensor_columns] = scaler.fit_transform(train_df[sensor_columns])
    test_df[sensor_columns] = scaler.transform(test_df[sensor_columns])
    
    return train_df, test_df, rul_df, scaler

# Load the data
train_df, test_df, rul_df, scaler = load_data()

# Sidebar for filters
st.sidebar.header("Filters")
selected_engine = st.sidebar.selectbox("Select Engine ID", train_df['Engine_ID'].unique())
selected_sensor = st.sidebar.selectbox("Select Sensor", [f'Sensor_{i}' for i in range(1, 20)])  # Only show 1-19

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Sensor Trends", "RUL Prediction", "Maintenance"])

with tab1:
    st.header("Engine Health Overview")
    engine_data = train_df[train_df['Engine_ID'] == selected_engine]
    
    # Current health status
    current_rul = engine_data['RUL'].iloc[-1]
    max_rul = engine_data['RUL'].max()
    health_percentage = (current_rul / max_rul) * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Engine ID", selected_engine)
    col2.metric("Current RUL", f"{current_rul} cycles")
    col3.metric("Health Status", f"{health_percentage:.1f}%")
    st.progress(int(health_percentage))
    
    # Historical degradation patterns
    st.subheader("Historical Degradation Patterns")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=engine_data, x='Cycle', y='RUL', ax=ax)
    ax.set_title(f"RUL Degradation for Engine {selected_engine}")
    st.pyplot(fig)

with tab2:
    st.header("Sensor Reading Trends")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=engine_data, x='Cycle', y=selected_sensor, ax=ax)
    ax.set_title(f"{selected_sensor} Trend for Engine {selected_engine}")
    st.pyplot(fig)
    
    st.subheader("Sensor Statistics")
    st.dataframe(engine_data[selected_sensor].describe().to_frame())

with tab3:
    st.header("Remaining Useful Life (RUL) Prediction")
    
    @st.cache_resource
    def get_model():
        X = train_df.drop(['Engine_ID', 'RUL'], axis=1)
        y = train_df['RUL']
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model
    
    model = get_model()
    current_engine_data = engine_data.iloc[-1:].drop(['Engine_ID', 'RUL'], axis=1)
    predicted_rul = model.predict(current_engine_data)[0]
    
    st.metric("Predicted RUL", f"{predicted_rul:.1f} cycles")
    
    # Feature importance
    st.subheader("Feature Importance")
    importance = pd.DataFrame({
        'Feature': train_df.drop(['Engine_ID', 'RUL'], axis=1).columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importance.head(10), x='Importance', y='Feature', ax=ax)
    st.pyplot(fig)

with tab4:
    st.header("Maintenance Planning")
    engine_rul = train_df.groupby('Engine_ID')['RUL'].last().sort_values()
    
    # Priority list
    st.subheader("Maintenance Priority")
    st.dataframe(engine_rul.reset_index().rename(columns={'RUL': 'Remaining Cycles'}))
    
    # Risk assessment
    st.subheader("Risk Distribution")
    risk_bins = [0, 50, 100, 200, float('inf')]
    risk_labels = ['Critical', 'High', 'Medium', 'Low']
    engine_risk = pd.cut(engine_rul, bins=risk_bins, labels=risk_labels)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    engine_risk.value_counts().sort_index().plot(kind='bar', ax=ax)
    st.pyplot(fig)