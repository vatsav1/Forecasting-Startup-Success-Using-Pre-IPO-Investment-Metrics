# Forecasting Startup Success Using Pre-IPO Investment Metrics

## Project Objective
The goal of this project is to develop a machine learning classifier that predicts whether a startup is likely to be successful (IPO, Acquired, or Operating) or unsuccessful (Closed/Inactive) based on its investment and funding history before going public. This has critical applications for venture capitalists, private equity firms, and investment analysts seeking to optimize portfolio decisions.

---

## Dataset
- **Source:** Crunchbase Startup Investment Data
- **File Used:** `ipo_cleaned.csv`
- **Records:** 54,294 startups
- **Features:** 35 columns including:
  - Company founding details
  - Country and market segment
  - Total funding, funding rounds, and types of funding (seed, venture, etc.)
  - Outcome status (IPO, Acquired, Operating, Closed)

---

## Key Steps
1. **Data Cleaning**
   - Removed irrelevant or text-heavy columns (e.g., `name`, `homepage_url`)
   - Handled missing values and formatted numeric fields

2. **Feature Engineering**
   - Derived `target` variable from status (Success = IPO, Acquired, Operating)
   - Created `funding_period` from first to last funding dates
   - Binned key numerical fields (e.g., `funding_total_usd`, `funding_rounds`)

3. **EDA**
   - Univariate and bivariate visualizations (with pie charts and bins)
   - Heatmap of correlations with the target variable

4. **Encoding & Resampling**
   - One-hot encoding on binned and categorical variables
   - Used SMOTE for class imbalance handling

5. **Modeling**
   - Logistic Regression (baseline)
   - Random Forest and XGBoost (advanced)
   - GridSearchCV for hyperparameter tuning

6. **Evaluation**
   - Accuracy, Recall, Precision, F1 Score, ROC-AUC
   - Confusion matrix and classification report
   - Visual insights from predicted vs. actual outcomes

---

## Final Outcome
A robust classification model capable of identifying high-potential startups from pre-IPO data with strong evaluation metrics, interpretable features, and visual insights. This supports smarter investment decisions and early identification of likely success stories in the startup ecosystem.

---

## Tools & Libraries
- Python, Pandas, NumPy
- Scikit-learn, XGBoost, imbalanced-learn
- Seaborn, Matplotlib, Plotly
- Jupyter Notebook

---

## Author
This project was developed by YSrivatsav. Contributions and improvements are welcome!

