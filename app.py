# ==============================================
# STREAMLIT APP: EV Energy Consumption Prediction
# ==============================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="EV Energy Consumption Prediction",
    layout="wide"
)

st.title("🔋 Energy Consumption Analysis & Prediction")
st.write("Mini Project using **Linear Regression & Random Forest**")

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("EV_Energy_Consumption_Dataset.csv")
    return df

df = load_data()

# ------------------------------------------------
# SHOW DATA
# ------------------------------------------------
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# ------------------------------------------------
# DATA PREPROCESSING
# ------------------------------------------------
if 'Vehicle_ID' in df.columns:
    df.drop(columns=['Vehicle_ID'], inplace=True)

if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df['Hour'] = df['Timestamp'].dt.hour
    df.drop(columns=['Timestamp'], inplace=True)

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# ------------------------------------------------
# EDA SECTION
# ------------------------------------------------
st.subheader("📊 Exploratory Data Analysis")

if st.checkbox("Show Data Info"):
    st.write(df.info())

if st.checkbox("Show Statistical Summary"):
    st.write(df.describe())

st.subheader("🔥 Correlation Heatmap")
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), cmap='viridis')
st.pyplot(plt)

# ------------------------------------------------
# MODEL TRAINING
# ------------------------------------------------
target = "Energy_Consumption_kWh"
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling for Linear Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)

# Random Forest
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# ------------------------------------------------
# MODEL EVALUATION
# ------------------------------------------------
st.subheader("📈 Model Evaluation")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Linear Regression")
    st.write("MAE:", mean_absolute_error(y_test, lr_pred))
    st.write("RMSE:", mean_squared_error(y_test, lr_pred, squared=False))
    st.write("R²:", r2_score(y_test, lr_pred))

with col2:
    st.markdown("### Random Forest")
    st.write("MAE:", mean_absolute_error(y_test, rf_pred))
    st.write("RMSE:", mean_squared_error(y_test, rf_pred, squared=False))
    st.write("R²:", r2_score(y_test, rf_pred))

# ------------------------------------------------
# VISUALIZATIONS
# ------------------------------------------------
st.subheader("📉 Actual vs Predicted")

fig1 = plt.figure(figsize=(6,4))
plt.scatter(y_test, lr_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Linear Regression")
st.pyplot(fig1)

fig2 = plt.figure(figsize=(6,4))
plt.scatter(y_test, rf_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Random Forest")
st.pyplot(fig2)

# ------------------------------------------------
# FEATURE IMPORTANCE
# ------------------------------------------------
st.subheader("⭐ Feature Importance")

rf_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.write(rf_importance.head(10))

plt.figure(figsize=(8,4))
plt.barh(rf_importance.head(10)["Feature"], rf_importance.head(10)["Importance"])
plt.gca().invert_yaxis()
st.pyplot(plt)

# ------------------------------------------------
# PREDICTION SECTION
# ------------------------------------------------
st.subheader("🔮 Predict Energy Consumption")

user_input = {}

for col in X.columns:
    user_input[col] = st.number_input(f"{col}", value=float(X[col].mean()))

input_df = pd.DataFrame([user_input])

input_scaled = scaler.transform(input_df)

lr_result = lr.predict(input_scaled)[0]
rf_result = rf.predict(input_df)[0]

st.success(f"🔹 Linear Regression Prediction: {lr_result:.2f} kWh")
st.success(f"🔹 Random Forest Prediction: {rf_result:.2f} kWh")