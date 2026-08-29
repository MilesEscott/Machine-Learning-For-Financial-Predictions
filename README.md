# SPY Next-Day Direction Prediction

A comparison of machine learning and baseline trading strategies for predicting the next-day
direction of the SPY ETF, built around a hand-rolled decision tree / random forest, a small
PyTorch neural network, and two simple rule-based baselines. Strategies are backtested and
ranked on total return, annualised return, annualised volatility, Sharpe ratio, and maximum
drawdown.

## Project structure

```
.
├── DataExploration.ipynb      # Pulls SPY price history and explores returns/volume/volatility
├── Features.ipynb             # Builds features and the train/validation/test split
├── BaselineStrategies.ipynb   # Buy-and-hold and momentum baselines
├── DecisionTree.ipynb         # Custom decision tree + random forest (bagged trees)
├── decisionTreeFunctions.py   # Gini-impurity decision tree / random forest implementation
├── NeuralNetwork.ipynb        # PyTorch feed-forward classifier
├── Comparison.ipynb           # Loads all predictions and evaluates/compares strategies
├── metrics.py                 # Return, volatility, Sharpe ratio, max drawdown helpers
├── Data/                      # Downloaded prices, engineered features, train/val/test CSVs
└── Predictions/               # Per-strategy prediction CSVs consumed by Comparison.ipynb
```

## Methodology

**Data.** Daily SPY OHLCV data is pulled via `yfinance` (~10 years of history). Adjusted close
prices are used so that dividend payouts don't distort returns.

**Features** (`Features.ipynb`), computed from daily closes/volume:
- 5-day return
- 20-day return
- 20-day rolling volatility (std. dev. of daily returns)
- Volume ratio (current volume vs. 20-day average volume)
- Target label: whether tomorrow's close is higher than today's close (`Tomorrow Up`)

Data is split chronologically to avoid lookahead bias:
- **Train:** 2016-01-01 → 2023-01-01
- **Validation:** 2023-01-01 → 2024-01-01
- **Test:** 2024-01-01 → 2026-01-01

**Baseline strategies** (`BaselineStrategies.ipynb`):
- *Buy and Hold* — always long.
- *Momentum* — long if the prior day's 20-day return was positive, flat otherwise.

**Decision tree / random forest** (`decisionTreeFunctions.py`, `DecisionTree.ipynb`) —
implemented from scratch rather than with a library:
- Candidate split thresholds are the 10th–90th percentiles of each feature.
- Splits are chosen by Gini impurity reduction.
- Several tree depths are grown and the depth that performs best on the validation set is
  selected for the final test.
- The random forest bags each tree on a random subset of the rows and a random pair of
  feature columns, then predicts by majority vote across trees.

**Neural network** (`NeuralNetwork.ipynb`) — a small PyTorch feed-forward classifier:
- Architecture: `4 → 32 → 8 → 1` with ReLU activations.
- Trained with `BCEWithLogitsLoss` and the Adam optimiser (lr `1e-3`) for 1000 epochs,
  validated each epoch on the validation set.

**Evaluation** (`metrics.py`, `Comparison.ipynb`) — each strategy's predictions are turned into
daily strategy returns (position × market return) and scored on:
- Total return
- Annualised return
- Annualised volatility
- (Raw) Sharpe ratio
- Maximum drawdown

## Results summary

Across the test period, Buy and Hold produced the highest total and annualised return, but the
other strategies reduced drawdown and volatility. The Neural Network and Momentum strategies
achieved better risk-adjusted returns (Sharpe ratio) than Buy and Hold, while the Decision Tree
and Random Forest underperformed — likely due to the limited feature set, a simple from-scratch
implementation, or the difficulty of predicting next-day direction from price/volume features
alone. Full figures are produced by `Comparison.ipynb`.

## Setup

```bash
pip install numpy pandas yfinance matplotlib torch
```

Python 3.13 was used during development; the notebooks should run on any recent Python 3
environment with the packages above.

## Running the pipeline

Run the notebooks in this order — each stage writes CSVs consumed by the next:

1. `DataExploration.ipynb` — optional, exploratory only
2. `Features.ipynb` — builds `Data/features.csv` and the train/validation/test splits
3. `BaselineStrategies.ipynb`, `DecisionTree.ipynb`, `NeuralNetwork.ipynb` — each writes its
   predictions to `Predictions/`
4. `Comparison.ipynb` — loads every prediction file and produces the final comparison table

## Notes

- The notebooks currently read/write using hardcoded absolute paths
  (`C:/Users/.../ml_project_cv/...`). Update these to match your local `Data/` and
  `Predictions/` directories (or switch to relative paths) before running.
- The decision tree and random forest are implemented from scratch for learning purposes
  rather than using `scikit-learn`.
