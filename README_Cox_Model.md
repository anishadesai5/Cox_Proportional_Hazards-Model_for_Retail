# Stratified Cox Proportional Hazards Model for Customer Repurchase Prediction

## 📋 Overview

This project implements a **Stratified Cox Proportional Hazards model** to predict customer repurchase behavior at the product (SKU) level using the UCI Online Retail dataset.

### What Problem Does This Solve?

In e-commerce/retail, businesses need to answer:
1. **Which customers are most likely to repurchase a specific product?**
2. **When will they repurchase it?**
3. **How can we target marketing efforts efficiently?**

Traditional classification models answer "will they buy?" but ignore timing. This Cox model predicts **both likelihood AND timing** of repurchase.

---

## 🎯 Key Features

### Stratified Approach
- **Separate baseline hazard for each product** (SKU)
  - Milk has ~7-day repurchase cycle
  - Shampoo has ~30-day cycle
  - Winter coats have ~365-day cycle
- **Shared customer behavior coefficients**
  - High-frequency customers behave similarly across all products

### Outputs
- **Risk scores** for each customer-product pair
- **Repurchase probability** at 30, 60, 90 days
- **Customer rankings** per product for targeted marketing
- **Demand forecasts** for inventory planning

---

## 📊 Dataset

### UCI Online Retail II Dataset

**Source:** 
- Kaggle: https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci
- UCI Repository: https://archive.ics.uci.edu/dataset/352/online+retail

**Description:**
- Transactions from UK-based online gift retailer
- Date range: 2009-12-01 to 2011-12-09
- ~1M transactions, 5,000+ customers, 4,000+ products

**Columns:**
- `InvoiceNo`: Transaction ID
- `StockCode`: Product/SKU identifier
- `Description`: Product name
- `Quantity`: Items purchased
- `InvoiceDate`: Transaction timestamp
- `UnitPrice`: Price per unit
- `CustomerID`: Unique customer ID
- `Country`: Customer location

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn lifelines scikit-learn openpyxl
```

### Step 1: Download Dataset

1. Go to https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci
2. Download `online_retail_II.xlsx`
3. Place in the same directory as the script

### Step 2: Run the Model

```bash
python stratified_cox_repurchase_model.py
```

### Step 3: View Results

Results will be saved to `/mnt/user-data/outputs/`:
- `repurchase_predictions.csv` - Customer rankings per product
- `survival_curves_product_*.png` - Visualization of survival curves

---

## 📈 How It Works

### Pipeline Overview

```
Raw Transactions
    ↓
Data Cleaning (remove cancellations, nulls)
    ↓
Create Survival Dataset
    ↓
    For each Customer × Product:
    - DURATION_DAYS: Days between purchases
    - EVENT: 1 = repurchased, 0 = censored
    ↓
Feature Engineering
    ↓
    Customer Features:
    - RECENCY: Days since first purchase
    - FREQUENCY: Total purchase count
    - MONETARY: Average spend
    - PRODUCT_FREQUENCY: Times bought this SKU
    - DAYS_SINCE_FIRST: Days since first bought this SKU
    ↓
Train Stratified Cox Model
    ↓
    strata=['StockCode']  ← Each product gets own baseline
    ↓
Predict & Rank Customers
    ↓
    For each product:
    - Risk score (relative hazard)
    - P(repurchase in 30/60/90 days)
    ↓
Business Insights & Recommendations
```

---

## 🔬 Model Details

### Stratified Cox Formula

For customer `i` buying product `s`:

```
h(t | X_i, product=s) = h₀ₛ(t) × exp(β₁×RECENCY + β₂×FREQUENCY + ... + βₚ×FEATURE_p)
                         ↑                ↑
                    Product-specific   Shared across products
                    baseline hazard    (customer behavior)
```

### Why Stratification?

| Scenario | Non-Stratified | Stratified (Our Model) |
|----------|----------------|------------------------|
| Milk purchase cycle | Uses same baseline as coats ❌ | Has its own 7-day baseline ✅ |
| Coat purchase cycle | Uses same baseline as milk ❌ | Has its own 365-day baseline ✅ |
| High-frequency customer effect | 2× baseline risk | 2× baseline risk (but product-specific baseline) |
| Prediction accuracy | Poor for diverse products | High for all products |

### Model Evaluation

**Concordance Index (C-index):**
- Measures how well model ranks customer repurchase timing
- Range: 0.5 (random) to 1.0 (perfect)
- Typical values: 0.65-0.75 for retail data

---

## 💡 Business Use Cases

### 1. Targeted Marketing Campaigns

**High Intent (>60% prob in 30 days):**
- Send gentle reminder email
- No discount needed (already primed to buy)

**Medium Intent (30-60% prob):**
- Offer 10-15% discount to accelerate purchase
- Personalized product recommendations

**Low Intent (<30% prob):**
- Skip for now (avoid annoying them)
- Try re-engagement campaign later

### 2. Inventory Planning

```python
# Expected repurchases for SKU_001 in next 30 days
expected_demand = predictions[
    (predictions['StockCode'] == 'SKU_001') & 
    (predictions['PROB_30D'] > 0.5)
].shape[0]

# Adjust inventory forecast
inventory_target = expected_demand * 1.2  # 20% buffer
```

### 3. Churn Prevention

```python
# Customers with unusually LOW risk for high-value products
at_risk = predictions[
    (predictions['RISK_SCORE'] < median_risk * 0.5) &
    (predictions['StockCode'].isin(high_value_products))
]

# Proactive retention: Send personalized offers
```

### 4. Product Bundling

```python
# Find customers likely to buy Product A but not Product B
candidates = predictions[
    (predictions['StockCode'] == 'PRODUCT_A') & (predictions['PROB_30D'] > 0.6)
]

candidates_b = predictions[
    (predictions['StockCode'] == 'PRODUCT_B') & (predictions['PROB_30D'] < 0.3)
]

bundle_targets = candidates.merge(candidates_b, on='CustomerID')
# Offer: "Buy Product A, get 20% off Product B"
```

---

## 📊 Example Output

### Top Customers for Product "85123A"

| CustomerID | RISK_SCORE | PROB_30D | PROB_60D | PROB_90D | Strategy |
|------------|------------|----------|----------|----------|----------|
| 12583 | 3.45 | 0.85 | 0.95 | 0.98 | Gentle reminder |
| 14646 | 2.89 | 0.72 | 0.88 | 0.93 | Light nudge |
| 15311 | 2.41 | 0.65 | 0.82 | 0.89 | Reminder email |
| 17850 | 1.87 | 0.48 | 0.68 | 0.79 | 10% discount |
| 13047 | 1.56 | 0.41 | 0.59 | 0.72 | 15% discount |

### Feature Importance

| Feature | Coefficient | Hazard Ratio | Interpretation |
|---------|-------------|--------------|----------------|
| PRODUCT_FREQUENCY | 0.35 | 1.42 | Each additional purchase of THIS product increases repurchase risk by 42% |
| FREQUENCY | 0.28 | 1.32 | Each additional purchase OVERALL increases risk by 32% |
| RECENCY | -0.15 | 0.86 | Each day since last purchase decreases risk by 14% (cooling off) |
| LOG_MONETARY | 0.12 | 1.13 | Higher spenders have 13% higher repurchase risk |

---

## 🔧 Customization

### Adjust Product Selection

```python
# Only use products with minimum 50 purchases
survival_df = create_repurchase_dataset(
    df_clean, 
    min_purchases=2, 
    min_product_purchases=50  # ← Adjust this
)
```

### Add More Features

```python
# In engineer_features() function, add:
survival_df['DAYS_BETWEEN_VISITS'] = ...  # Average time between visits
survival_df['CATEGORY_AFFINITY'] = ...     # % of purchases in this category
survival_df['DISCOUNT_SENSITIVITY'] = ...  # How often buys on discount
```

### Change Time Horizons

```python
# Predict at different time points
time_horizons = [7, 14, 30, 60, 90, 180]  # days

for horizon in time_horizons:
    if horizon in survival_func.index:
        product_df[f'PROB_{horizon}D'] = 1 - survival_func.loc[horizon].values
```

---

## 📚 Key Concepts

### Survival Analysis Terms

- **Survival Function S(t):** Probability of NOT repurchasing by time t
- **Hazard Function h(t):** Instantaneous risk of repurchasing at time t
- **Censoring:** Customer hasn't repurchased YET (not "never")
- **Event:** Repurchase occurred
- **Duration:** Time between purchases (or time since last purchase if censored)

### Cox Model Terms

- **Baseline Hazard h₀(t):** Default repurchase risk over time (non-parametric)
- **Partial Hazard / Risk Score:** Relative risk multiplier based on features
- **Hazard Ratio exp(β):** How much a feature changes repurchase risk
- **Concordance Index:** Model's ability to correctly rank repurchase timing

---

## 🎓 Learning Resources

### Papers & Articles
1. Cox, D. R. (1972). "Regression Models and Life-Tables"
2. Chen, D. et al. (2012). "Data mining for online retail: RFM model-based customer segmentation"

### Lifelines Documentation
- Tutorial: https://lifelines.readthedocs.io/
- Cox PH API: https://lifelines.readthedocs.io/en/latest/fitters/regression/CoxPHFitter.html

### Related GitHub Repos
- archd3sai/Customer-Survival-Analysis-and-Churn-Prediction
- archie-cm/Churn-Analysis-Ecommerce-Customer

---

## 🤝 Contributing

Improvements welcome! Areas for enhancement:
- Time-varying covariates (e.g., seasonality)
- Deep learning extensions (DeepSurv)
- A/B test framework integration
- Real-time prediction API

---

## 📄 License

MIT License - Free to use and modify

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the dataset
- lifelines library by Cameron Davidson-Pilon
- Survival analysis research community

---

## 📧 Questions?

Open an issue or contact the maintainer.

**Happy modeling! 🚀**
