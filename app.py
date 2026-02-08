import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Set page configuration
st.set_page_config(page_title="Biodiesel Yield Prediction", layout="wide")

# Title and User Info
st.title("Biodiesel Yield Prediction Model")
st.markdown("### Developed by: **Bhupesh Danewa**")
st.markdown("#### College: **Maulana Azad National Institute of Technology**")

@st.cache_data
def load_data():
    # Assuming the file is in the same directory as the app
    file_path = "Fully_Preprocessed_Compiled_Dataset.xlsx"
    try:
        df = pd.read_excel(file_path)
        # Sanitize column names as per notebook logic
        df.columns = df.columns.str.replace(r'[^\w]', '_', regex=True)
        return df
    except FileNotFoundError:
        st.error(f"File not found: {file_path}. Please make sure the dataset is in the same directory as this app.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

if df is not None:
    # Prepare Data
    # Identifying features and target based on notebook analysis
    # target is the last column 'Biodiesel_yield___'
    target_col = df.columns[-1]
    feature_cols = df.columns[:-1]
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Split Data (70% train, 15% val, 15% test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)
    
    @st.cache_resource
    def train_model():
        # ANN Model architecture from notebook
        model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            BatchNormalization(),
            Dropout(0.2),

            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),

            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.1),

            Dense(16, activation='relu'),
            Dense(1)  # Output layer
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        early_stop = EarlyStopping(patience=30, restore_best_weights=True, monitor='val_loss')
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )
        return model

    with st.spinner("Training Model... This might take a moment."):
        model = train_model()
    
    st.success("Model Trained Successfully!")

    # Calculate metrics on test set to show model performance
    y_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    st.sidebar.markdown(f"**Model Performance (Test R²):** {test_r2:.4f}")

    # Sidebar for inputs
    st.sidebar.header("Input Parameters")
    
    def user_input_features():
        inputs = {}
        for col in feature_cols:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            mean_val = float(df[col].mean())
            
            # Clean up column name for display
            # Example: Molar_Ratio_MeOH__Oil_ -> Molar Ratio MeOH Oil
            display_name = col.replace('_', ' ').strip()
            
            inputs[col] = st.sidebar.slider(display_name, min_val, max_val, mean_val)
        return pd.DataFrame(inputs, index=[0])

    input_df = user_input_features()

    # Main panel predictions
    st.subheader("User Input Parameters")
    st.dataframe(input_df)

    if st.button("Predict Yield"):
        prediction = model.predict(input_df)
        st.markdown("---")
        st.subheader("Predicted Biodiesel Yield")
        st.info(f"The predicted yield is: **{prediction[0][0]:.4f}**")

else:
    st.warning("Please upload the dataset or ensure it is in the correct path.")
