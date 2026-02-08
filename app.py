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
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(page_title="Biodiesel Yield Prediction", layout="wide")

# Title and User Info
st.title("Biodiesel Yield Prediction Model")
st.markdown("### Developed by: **Bhupesh Danewa**")
st.markdown("#### College: **Maulana Azad National Institute of Technology**")

def remove_outliers_iqr(df):
    """
    Remove outliers from the dataframe using the IQR method.
    Replicates logic from data-preprocessing.ipynb
    """
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include='number').columns
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

@st.cache_data
def load_and_preprocess_data():
    # Load Raw Data
    file_path = "Compiled Dataset.xlsx"
    try:
        df = pd.read_excel(file_path)
        # Drop rows with missing values
        df = df.dropna()
        
        # Sanitize column names
        df.columns = df.columns.str.replace(r'[^\w]', '_', regex=True)
        
        # Outlier Removal
        df_clean = remove_outliers_iqr(df)
        
        return df_clean
    except FileNotFoundError:
        st.error(f"File not found: {file_path}. Please make sure the dataset is in the same directory as this app.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load formatted data (outliers removed but not scaled yet)
df = load_and_preprocess_data()

if df is not None:
    # Define Features and Target
    target_col = df.columns[-1]
    feature_cols = df.columns[:-1]
    
    X = df[feature_cols]
    y = df[[target_col]] # Keep as 2D DataFrame for scaler
    
    # Initialize Scalers
    # We must fit scalers here so they can be reused for user input
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    # Fit and Transform Data
    X_scaled = pd.DataFrame(feature_scaler.fit_transform(X), columns=X.columns)
    y_scaled = pd.DataFrame(target_scaler.fit_transform(y), columns=y.columns)
    
    # Split Data (70% train, 15% val, 15% test)
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_scaled, test_size=0.30, random_state=42)
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

    with st.spinner("Training Model on preprocessed dataset... This might take a moment."):
        model = train_model()
    
    st.success("Model Trained Successfully!")

    # Sidebar for inputs (Using RAW values from the Cleaned Dataframe for sliders)
    st.sidebar.header("Input Parameters")
    
    def user_input_features():
        inputs = {}
        for col in feature_cols:
            # We use the cleaned dataframe statistics for sliders to ensure valid ranges
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            mean_val = float(df[col].mean())
            
            display_name = col.replace('_', ' ').strip()
            
            inputs[col] = st.sidebar.slider(display_name, min_val, max_val, mean_val)
        return pd.DataFrame(inputs, index=[0])

    input_df = user_input_features()

    # Main panel predictions
    st.subheader("User Input Parameters")
    st.dataframe(input_df)

    if st.button("Predict Yield"):
        # 1. Scale User Input
        input_scaled = feature_scaler.transform(input_df)
        
        # 2. Get Prediction (Single Value)
        prediction_scaled = model.predict(input_scaled)
        
        # 3. Inverse Transform Target
        prediction_raw = target_scaler.inverse_transform(prediction_scaled)
        
        # 4. Clip result to max 100%
        final_yield = min(prediction_raw[0][0], 100.0)
        final_yield = max(final_yield, 0.0) # Ensure no negative yield either
        
        st.markdown("---")
        st.subheader("Predicted Biodiesel Yield")
        st.info(f"The predicted yield is: **{final_yield:.2f}%**")

else:
    st.warning("Please upload the dataset or ensure 'Compiled Dataset.xlsx' is in the correct path.")

