# End-to-End Demand Forecasting: M5 Accuracy Challenge

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=for-the-badge&logo=pandas)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-white?style=for-the-badge&logo=lightgbm)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

A large-scale demand forecasting and inventory optimization system built on the **M5 Forecasting - Accuracy** dataset. This project implements a full machine learning pipeline to predict daily unit sales for thousands of product-store combinations across various retail categories.

---

## 🎯 Project Objective
The goal is to accurately forecast daily sales (SKU × Store level) for the next **28 days**. This information is critical for:
- **Inventory Optimization**: Reducing stockouts and minimizing overstock.
- **Supply Chain Efficiency**: Managing lead times and logistical planning.
- **Scalability**: Handling over 30,000 unique time series simultaneously.

---

## 🏗️ Project Architecture

### 1. Business Problem Formulation
- **Granularity**: Product (SKU) × Store × Date.
- **Horizon**: 28-day window.
- **Constraints**: High percentage of zero-sales days (intermittent demand) and high volatility.

### 2. Data Preparation & Engineering
- **Melted Data**: Transformed sales data from wide to long format for time-series modeling.
- **Feature Sets**:
    - **Temporal**: Year, Month, Weekday, Week of Year, Event flags (Snap days).
    - **Lag Features**: Historical sales at specific intervals (e.g., Lag 7, 28).
    - **Rolling Window**: Moving averages and standard deviations (7-day and 28-day windows) to capture trends.
    - **Momentum**: Ratios of short-term to long-term sales patterns.

### 3. Modeling: LightGBM
Used a **Gradient Boosting Decision Tree (GBDT)** approach via LightGBM, optimized for:
- Speed and low memory usage.
- Handling sparse data and categorical features natively.
- Early stopping to prevent overfitting.

---

## 📂 Project Structure
```text
.
├── m5_demand_forecasting.ipynb # Core notebook (Preprocessing -> Model)
├── .gitignore                  # Prevents large CSVs from being tracked
└── README.md                   # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook or Lab

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/khushiarora0208/end-to-end-demand-forecasting.git
   cd end-to-end-demand-forecasting
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy lightgbm scikit-learn matplotlib seaborn
   ```

3. **Download the Dataset**:
   The data files are too large for GitHub tracking and are ignored via `.gitignore`. Download the dataset directly from Kaggle:
   - **Link**: [Kaggle M5 Forecasting - Accuracy Data](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)
   - Unzip and place the `.csv` files inside the `m5-forecasting-accuracy/` folder.

---

## 📈 Performance & Evaluation
The model is evaluated using **Root Mean Squared Error (RMSE)** and **Mean Absolute Error (MAE)**. 
- **Validation RMSE**: ~2.01 (Current Iteration)
- **Validation MAE**: ~1.05 (Current Iteration)

### Feature Importance
The most influential features typically include:
- **Lag 7**: Sales from exactly one week ago.
- **Rolling Mean 28**: Long-term trend indicator.
- **Snapshot (Event)**: Whether a SNAPs/Holiday occurred.

---

## 🤝 Acknowledgments
- **Kaggle**: For providing the M5 Forecasting - Accuracy challenge data.
- **Walmart**: For the real-world dataset.
- **The Open Source Community**: For the powerful tools (LightGBM, Pandas) that make large-scale forecasting accessible.
