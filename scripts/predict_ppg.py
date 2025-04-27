
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load NBA data (example: using Kaggle's dataset)
df = pd.read_csv('nba_player_stats.csv')  # Make sure the CSV is in the same directory

# Feature selection
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
plt.savefig('/mnt/data/nba_ppg_predictor/plots/feature_importance.png')
plt.show()
