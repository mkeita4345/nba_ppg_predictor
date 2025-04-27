
# NBA Points Per Game Prediction

This project aims to predict the points per game (PPG) for NBA players using machine learning techniques, specifically a Random Forest Regressor.

## Files and Structure:
- `scripts/predict_ppg.py`: Contains the script to load NBA data, train a machine learning model, and visualize the results.
- `plots/feature_importance.png`: A plot showing the importance of each feature in the model.
- `README.md`: Overview and instructions for the project.

## Setup Instructions:
1. Install necessary libraries:
    ```bash
    pip install pandas scikit-learn matplotlib seaborn
    ```

2. Ensure you have the `nba_player_stats.csv` file in the project directory.

3. Run the `predict_ppg.py` script to train the model and view feature importance.

## Future Work:
- Implement a web dashboard to allow team managers and fans to interact with the model and make predictions.
