import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.chdir('C:/Users/Moctar.Keita/nba_ppg_predictor')  # Set working directory to your project folder
# Define file paths
csv_file_path = 'nba_data_processed.csv'  # Relative path to CSV file
plots_folder = 'plots'  # Folder where the plot image will be saved



# Check if the plots folder exists, if not, create it
if not os.path.exists(plots_folder):
    os.makedirs(plots_folder)

# Try loading the CSV file
try:
    df = pd.read_csv(csv_file_path)  # Load the CSV file
    print("CSV file loaded successfully.")
except Exception as e:
    print(f"Error loading CSV file: {e}")
    exit()  # Stop execution if there's an error loading the file

# Check the first few rows of the dataframe
print(df.head())  # Optional: To ensure the data is loaded correctly

# Feature selection (ensure the column names match your dataset)
features = ['age', 'height', 'weight', 'games_played', 'field_goal_percentage', 'three_point_percentage', 'assists', 'rebounds']
X = df[features]
y = df['points_per_game']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict points per game for the test set
y_pred = model.predict(X_test)

# Plot the feature importances
importances = model.feature_importances_
indices = pd.Series(importances, index=features).sort_values(ascending=False)
sns.barplot(x=indices, y=indices.index)
plt.title("Feature Importance in Predicting NBA Points Per Game")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()

# Save the plot to the 'plots' folder
plot_file_path = os.path.join(plots_folder, 'feature_importance.png')
plt.savefig(plot_file_path)
plt.show()

print(f"Plot saved to {plot_file_path}")
