#Apple stock trend prediction

##BLUF 
This project trains ML models to predict whether Apple stock price will go 
up or down the next day based on historical data and technical indicators. 
Models used in this project include: Logistic Regression, Random Forest, 
XGBoost, and CATBoost evaluated with time-series cross validation. 

## Project Overview

- Load raw Apple data from `data/APL_data` (tab-separated).
- Convert it to `data/APL_data.csv`.
- Clean columns and construct a `date` column.
- Compute log returns and define a binary `trend` target:
  - 1 if next-day log return > 0
  - 0 otherwise
- Engineer technical features via `FeatureEngineeringTransformer`.
- Train and tune models with `TimeSeriesSplit` and `RandomizedSearchCV`.
- Evaluate performance and save best pipelines.

Main notebook: `Apple.ipynb`.

##Data
Raw data: apple-returns-ml/data/APL/data
CSV: apple-returns-ml/data/APL_data.csv

## How to Run

From the project root:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm 
catboost imbalanced-learn joblib
jupyter notebook [Apple.ipynb](http://_vscodecontentref_/0)
