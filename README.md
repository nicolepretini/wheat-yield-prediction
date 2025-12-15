# Wheat Yield Predictability Benchmark (FAOSTAT)

This project evaluates how much national wheat yield variability can be explained
using only historical trends and country identity, based on FAOSTAT data.

Rather than building an operational forecasting model, the goal is to establish
a transparent baseline for country-level wheat yield predictability under minimal
information assumptions.

## Data

FAOSTAT crop statistics (1961–2021), filtered to:
- Crop: Wheat
- Element: Yield
- Spatial scale: Country

Final dataset (504 observations) includes:
- `Year`
- `Region` (country)
- `Yield` (kg/ha, as reported by FAOSTAT)

## Methods

### Feature set
- `Year` (numeric)
- `Region` (categorical, one-hot encoded)

No climate, soil, or management variables are used.

### Models
- Linear Regression (baseline)
- Random Forest Regressor (nonlinear benchmark)

### Validation strategy

To avoid temporal leakage, model performance is evaluated using a **rolling
10-year forward-chaining validation scheme**:

- Models are trained on all data up to year *T*
- Tested on the subsequent decade (*T+1* to *T+10*)
- This process is repeated across multiple decades

This approach provides a more realistic assessment than random train/test splits,
which tend to inflate performance in time-structured data.

### Metrics
- RMSE (kg/ha)
- R² (coefficient of determination)

## Results

Mean performance across evaluation decades:

| Model             | RMSE (kg/ha) | R²   |
|------------------|-------------:|:----:|
| Linear Regression | ~634         | 0.87 |
| Random Forest     | ~480         | 0.92 |

Performance varies across decades, with higher predictability in more recent
periods, reflecting technological convergence and more stable yield trends.

For reference, random train/test splits yield higher apparent performance
(R² ≈ 0.96), highlighting the importance of time-aware validation.

## Interpretation

These results indicate that a substantial fraction of national wheat yield
variability can be explained by long-term trends and country-specific effects
alone.

However, performance decreases under forward-chaining validation, emphasizing
the limits of purely historical extrapolation and the need for climate and
management variables for true predictive applications.

This benchmark provides a transparent reference point against which more
mechanistic or data-rich yield models can be compared.

## Tech stack
- Python
- pandas
- scikit-learn
- matplotlib
