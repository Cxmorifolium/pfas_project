import numpy as np
import pandas as pd
import ast
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
import os

# Step 1: Load and Preprocess Data
df = pd.read_csv("/home/atupulazi/personal_projects/AI Practicum-20241107T183353Z-001/AI-Practicum/Files/GeoText.2010-10-12/full_text.txtfile_with_embeddings.csv")

# Clean and process CLS_EMBEDDING
df['CLS_EMBEDDING'] = df['CLS_EMBEDDING'].str.replace(r'^\s*,', '', regex=True) #only removes spaces if they’re followed by a comma at the start of string
df['CLS_EMBEDDING'] = df['CLS_EMBEDDING'].str.replace(r'\s+', ',', regex=True) #replaces all spaces with commas
df['CLS_EMBEDDING'] = df['CLS_EMBEDDING'].apply(lambda x: f"[{x.strip('[] ')}]" if not x.startswith('[') else x)

# Convert embeddings to NumPy arrays
def safe_literal_eval(val):
    try:
        return np.array(ast.literal_eval(val))
    except (ValueError, SyntaxError):
        return np.nan

df['CLS_EMBEDDING'] = df['CLS_EMBEDDING'].apply(safe_literal_eval)
df = df.dropna(subset=['CLS_EMBEDDING'])  # Drop rows with invalid embeddings (NULL or NaN)

# Extract features (X) and labels (y)
X = np.stack(df['CLS_EMBEDDING'].values)
y = df['STATE'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Define Random Forest and GridSearch Parameters
rf_model = RandomForestClassifier(random_state=42, class_weight="balanced")

param_grid = {
    "n_estimators": [100, 200],# Number of trees
    "max_depth": [10, None], # Maximum depth of trees
    "min_samples_split": [2, 5], # Minimum samples to split a node
    "min_samples_leaf": [1, 2], # Minimum samples for a leaf node
    "max_features": ['sqrt', 'log2'] # Number of features to consider at each split
}

# Step 3: Train with GridSearchCV and Add Partial Save
if os.path.exists("grid_search_partial.pkl"):
    print("Loading partially completed GridSearchCV...")
    grid_search = joblib.load("grid_search_partial.pkl")
else:
    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        scoring="accuracy",
        cv=4,  # Reduced for speed
        n_jobs=-1,  # Use all CPU cores
        verbose=2  # Show detailed progress
    )

try:
    grid_search.fit(X_train, y_train)
except KeyboardInterrupt:
    # Save the partially completed GridSearchCV if interrupted
    print("Interrupt detected, saving progress...")
    joblib.dump(grid_search, "grid_search_partial.pkl")
    print("Progress saved. You can resume later.")
    raise  # Re-raise exception for debugging or to stop execution

# Save the best model if GridSearchCV completes
best_model = grid_search.best_estimator_
joblib.dump(best_model, "optimized_random_forest_model.pkl")
print("Grid Search complete. Best model saved.")

# Print best parameters and cross-validated training accuracy
print("Best Parameters:", grid_search.best_params_)
print("Best Training Accuracy (CV):", grid_search.best_score_)

# Step 4: Evaluate the Best Model
y_pred = best_model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=1))

# Save predictions for analysis
results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
results.to_csv("optimized_predictions.csv", index=False)
print("Predictions saved as optimized_predictions.csv")

# Step 5: Visualize Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
disp.plot(cmap='viridis')
plt.show()

# Optional: High-PFAS states analysis
high_pfas_states = ['Michigan', 'New York', 'California']  # Example high-PFAS states
high_pfas_df = df[df['STATE'].isin(high_pfas_states)]

X_high_pfas = np.stack(high_pfas_df['CLS_EMBEDDING'].values)
y_high_pfas = high_pfas_df['STATE'].values

y_pred_high_pfas = best_model.predict(X_high_pfas)
print("High-PFAS Accuracy:", accuracy_score(y_high_pfas, y_pred_high_pfas))
print(classification_report(y_high_pfas, y_pred_high_pfas, zero_division=1))
