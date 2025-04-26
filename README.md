# FUTURE_ML_02
# 📈 Microsoft (MSFT) Stock Price Prediction using LSTM

---

## 🧾 Project Overview

This project focuses on forecasting the **closing stock prices** of **Microsoft Corporation (MSFT)** using a **Long Short-Term Memory (LSTM)** deep learning model.  
It demonstrates time series modeling, evaluation, and visualization based on historical stock data.

---

## 🛠️ Technologies and Libraries Used

- Python
- Pandas, NumPy — Data processing
- Matplotlib, Seaborn — Data visualization
- TensorFlow (Keras) — LSTM model development
- Scikit-learn — Data scaling and evaluation metrics

---

## 📂 Dataset Details

- File: `MSFT.csv`
- Columns: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`
- Target Variable: `Close`
- Total Records: *(as per dataset size)*

---

## 📋 Steps Performed

1. Loaded and cleaned the Microsoft stock dataset.
2. Used only the `Close` price for prediction.
3. Scaled data using **MinMaxScaler** to normalize between [0, 1].
4. Created sequences of **60 previous days** to predict the next day's price.
5. Split data into **80% training** and **20% testing**.
6. Built and trained an **LSTM model**.
7. Evaluated model performance.
8. Visualized predictions and errors.

---

## 🧠 Model Architecture

| Layer           | Parameters |
|-----------------|------------|
| LSTM Layer 1    | 50 units, return_sequences=True |
| Dropout Layer   | 20% |
| LSTM Layer 2    | 50 units, return_sequences=False |
| Dropout Layer   | 20% |
| Dense Layer     | 1 neuron (final output) |

- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)
- **Epochs**: 50
- **Batch Size**: 32

---

## 📊 Evaluation Metrics

| Metric            | Value |
|-------------------|-------|
| MAE (Mean Absolute Error) | 2.83 |
| RMSE (Root Mean Squared Error) | 4.36 |
| R² Score          | 0.99 |
| MAPE (Mean Absolute Percentage Error) | 2.96% |
| Accuracy          | 97.06% |

---

## 📈 Visualizations

- Actual vs Predicted Prices (Full Test Data)
- Error Distribution (Histogram + KDE)
- Zoomed Comparison (First 100 Predictions)

---

## ✅ Key Outcomes

- Successfully built an LSTM-based model to forecast MSFT stock prices.
- Achieved reasonable accuracy and good prediction trends.
- Provided visual insights into model performance.

---

## 📌 Future Improvements

- Try different sequence lengths (e.g., 90, 120 days).
- Hyperparameter tuning (units, layers, learning rate).
- Use multivariate inputs (Volume, Open, High, Low prices).
- Deploy the model for real-time stock forecasting.

---

# 🚀 Conclusion

This project showcases how **Deep Learning (LSTM)** can be effectively applied to **time series forecasting** problems such as stock price prediction.

---



