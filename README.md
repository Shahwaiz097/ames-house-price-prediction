# Ames House Price Prediction

This is a simple machine-learning project for predicting house prices.

The project was converted from the original notebook into a clean GitHub structure.

## Project structure

```text
ames-house-price-prediction/
│
├── src/
│   └── train.py
│
├── data/
│   └── house_prices_sample.csv
│
├── notebooks/
│   └── original_notebook.ipynb
│
├── outputs/
│   └── metrics and plots will be saved here
│
├── models/
│   └── trained model will be saved here
│
├── README.md
└── requirements.txt
```

## Dataset

The script first checks for:

```text
data/house_prices.csv
```

If this file is missing, it tries to download the full Ames House Prices dataset from OpenML.

A small sample file is included:

```text
data/house_prices_sample.csv
```

This sample file is only included so the project can still run when internet access is not available.

## How to run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python src/train.py
```

## What the script does

1. Loads the house prices data
2. Splits the data into training and testing sets
3. Handles missing values
4. Encodes categorical columns
5. Trains a machine-learning model
6. Saves evaluation results in `outputs/`
7. Saves the trained model in `models/`

## Output files

After running the project, these files will be created:

```text
outputs/metrics.csv
outputs/predictions.csv
outputs/actual_vs_predicted.png
models/house_price_model.pkl
```

## Model used

The project uses `HistGradientBoostingRegressor` from scikit-learn.
