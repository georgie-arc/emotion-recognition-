import tensorflow as tf
import os
import sys

# Path to your model
model_path = r'C:\Users\barig\OneDrive\Documents\opencv\.vscode\facialemotionmodel.h5'

# Check if the file exists
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    sys.exit()

# Load and summarize the model
try:
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.\n")

    print("Model Summary:")
    model.summary()

    # Extra flush to ensure it prints in some IDEs
    sys.stdout.flush()

except Exception as e:
    print(f"Error loading model: {e}")
