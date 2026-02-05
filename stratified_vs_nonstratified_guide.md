# Stratified vs Non-Stratified Cox Models: Complete Guide

## Visual Comparison

```
┌─────────────────────────────────────────────────────────────────────┐
│                     NON-STRATIFIED COX MODEL                        │
│                   (Standard Approach - NOT Ideal)                   │
└─────────────────────────────────────────────────────────────────────┘

Formula: h(t|X) = h₀(t) × exp(β₁X₁ + β₂X₂ + ...)
                  ↑
         ONE baseline hazard for ALL products

┌─────────────────────────────────────────────────────────────────────┐
│             ALL Products Share Same Baseline Hazard                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Hazard                                                            │
│     ▲                                                               │
│     │        ╭─╮                                                    │
│     │       ╱   ╲                                                   │
│     │      ╱     ╲___                                               │
│     │     ╱          ╲___                                           │
│     │____╱_______________╲_____________________▶ Days               │
│     0    30    60    90    120   150                                │
│                                                                     │
│  ❌ PROBLEM: This doesn't make sense!                               │
│     • Milk repurchase peak: ~7 days                                 │
│     • Shampoo repurchase peak: ~30 days                             │
│     • Winter coat repurchase peak: ~365 days                        │
│     • They DON'T share the same timing pattern!                     │
└─────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                     STRATIFIED COX MODEL                            │
│                 (Our Approach - BETTER!)                            │
└─────────────────────────────────────────────────────────────────────┘

Formula: h(t|X,s) = h₀ₛ(t) × exp(β₁X₁ + β₂X₂ + ...)
                    ↑
         SEPARATE baseline per product stratum 's'

┌─────────────────────────────────────────────────────────────────────┐
│         Product 1: Milk (fast repurchase cycle)                     │
├─────────────────────────────────────────────────────────────────────┤
│   Hazard                                                            │
│     ▲                                                               │
│     │  ╭╮                                                           │
│     │ ╱  ╲                                                          │
│     │╱    ╲___                                                      │
│     │─────────╲_______________________________▶ Days                │
│     0   7   14   21   28   35                                       │
│                                                                     │
│  ✅ Peak at ~7 days (weekly grocery shopping)                       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│         Product 2: Shampoo (medium repurchase cycle)                │
├─────────────────────────────────────────────────────────────────────┤
│   Hazard                                                            │
│     ▲                                                               │
│     │           ╭───╮                                               │
│     │          ╱     ╲                                              │
│     │         ╱       ╲___                                          │
│     │________╱____________╲___________________▶ Days                │
│     0    30    60    90   120                                       │
│                                                                     │
│  ✅ Peak at ~30 days (monthly personal care)                        │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│         Product 3: Winter Coat (slow repurchase cycle)              │
├─────────────────────────────────────────────────────────────────────┤
│   Hazard                                                            │
│     ▲                                                               │
│     │                                          ╭╮                   │
│     │                                         ╱  ╲                  │
│     │________________________________________╱____╲_▶ Days          │
│     0    90   180   270   365   455   545                           │
│                                                                     │
│  ✅ Peak at ~365 days (annual replacement)                          │
└─────────────────────────────────────────────────────────────────────┘

BUT: Customer features (β coefficients) are SHARED across all products!
• High-frequency customer: 2× risk for ALL products
• High-spend customer: 1.5× risk for ALL products
```

---

## Detailed Mathematical Comparison

### Non-Stratified Model

**Model:**
```
h(t | X_customer) = h₀(t) × exp(β₁×frequency + β₂×monetary + ...)
```

**Example Predictions:**

Customer A (High frequency: 20 purchases)
- Milk prediction uses h₀(t)
- Shampoo prediction uses h₀(t)  ← SAME baseline
- Coat prediction uses h₀(t)      ← SAME baseline

**Problem:** All products forced to share repurchase timing!

---

### Stratified Model (Our Approach)

**Model:**
```
h(t | X_customer, product) = h₀_product(t) × exp(β₁×frequency + β₂×monetary + ...)
```

**Example Predictions:**

Customer A (High frequency: 20 purchases)
- Milk prediction uses h₀_milk(t)     ← Peaks at 7 days
- Shampoo prediction uses h₀_shampoo(t) ← Peaks at 30 days
- Coat prediction uses h₀_coat(t)       ← Peaks at 365 days

**Solution:** Each product has its own timing, but customer behavior affects all similarly!

---


## Concrete Example with Numbers

### Scenario
- **Customer A:** Frequent shopper (20 purchases/year)
- **Customer B:** Occasional shopper (4 purchases/year)
- **Products:** Milk, Shampoo, Winter Coat

### Non-Stratified Model (WRONG)

```
                    Day 7    Day 30   Day 365
Customer A (all)    0.15     0.08     0.01     ← SAME for all products!
Customer B (all)    0.08     0.04     0.005
```

❌ This says Customer A has same repurchase hazard for milk and coat at day 7!
❌ Coat repurchase at day 7? Doesn't make sense!

### Stratified Model (CORRECT)

```
Product: Milk (peaks day 7)
                    Day 7    Day 30   Day 365
Customer A          0.30     0.05     0.00
Customer B          0.15     0.025    0.00

Product: Shampoo (peaks day 30)
                    Day 7    Day 30   Day 365
Customer A          0.05     0.25     0.01
Customer B          0.025    0.125    0.005

Product: Winter Coat (peaks day 365)
                    Day 7    Day 30   Day 365
Customer A          0.00     0.01     0.20
Customer B          0.00     0.005    0.10
```

✅ Each product has its own realistic timing pattern!
✅ Customer A always has 2× the risk of Customer B (proportional hazards)
✅ But absolute timing differs by product

---

## When to Use Stratification

### ✅ USE Stratified Model When:

1. **Fundamentally different baseline risks**
   - Products with very different repurchase cycles
   - Customer segments with different churn patterns
   - Geographic regions with different seasonality

2. **Proportional hazards assumption violated**
   - Survival curves cross over time
   - Hazard ratios change significantly over time
   - Test shows p < 0.05 for PH assumption

3. **You care about relative effects, not absolute baselines**
   - Want to know: "Does high frequency increase repurchase risk?"
   - Don't care: "What's the absolute baseline risk for Product X vs Y?"

4. **Stratifying variable is nuisance factor**
   - Product is just context, not the focus
   - Focus is on customer behavior patterns
   - Product differences are known and accepted

### ❌ DON'T Use Stratification When:

1. **You want to estimate the effect of the stratifying variable**
   - Want coefficient for Product Type
   - Need to compare absolute risk between products
   - Want to predict for NEW products not in training

2. **Very sparse data per stratum**
   - <20 events per product
   - Can't reliably estimate baseline per stratum
   - Better to use product as covariate instead

3. **Baselines are actually similar**
   - All products have similar repurchase timing
   - Stratification adds complexity without benefit

---

## Code Implementation Comparison

### Non-Stratified (Standard)

```python
from lifelines import CoxPHFitter

cph = CoxPHFitter()
cph.fit(
    df,
    duration_col='DURATION_DAYS',
    event_col='EVENT'
    # No strata parameter
)

# Result: One baseline h₀(t) for everyone
```

### Stratified (Our Approach)

```python
from lifelines import CoxPHFitter

cph = CoxPHFitter()
cph.fit(
    df,
    duration_col='DURATION_DAYS',
    event_col='EVENT',
    strata=['StockCode']  # 🔑 KEY DIFFERENCE
)

# Result: Separate baseline h₀_product(t) per product
# But shared β coefficients
```

---

## Interpretation Differences

### Non-Stratified Output

```
Coefficient Summary:
                    coef  exp(coef)  p-value
FREQUENCY           0.25      1.28    0.001
PRODUCT_TYPE_MILK   0.80      2.23    0.000  ← CAN estimate this
PRODUCT_TYPE_COAT  -0.50      0.61    0.010  ← CAN estimate this

Concordance Index: 0.68
```

**Interpretation:**
- Frequent customers have 28% higher risk
- **Milk customers have 2.23× risk vs baseline**
- **Coat customers have 0.61× risk vs baseline**

### Stratified Output

```
Coefficient Summary:
                    coef  exp(coef)  p-value
FREQUENCY           0.25      1.28    0.001
                                            ← CANNOT estimate PRODUCT effect
                                            ← (it's in the baseline!)

Concordance Index: 0.74  ← Usually HIGHER!

Number of strata: 3 (Milk, Shampoo, Coat)
```

**Interpretation:**
- Frequent customers have 28% higher risk **within each product**
- Each product has its own baseline (not estimated as coefficient)
- Better concordance because product timing differences captured

---

## Prediction Differences

### Scenario: Predict for Customer A (high frequency)

**Non-Stratified:**
```python
# Predict for Customer A buying milk
customer_a_milk = df_milk[df_milk['CustomerID'] == 'A']
survival_milk = cph.predict_survival_function(customer_a_milk)

# Predict for Customer A buying coat  
customer_a_coat = df_coat[df_coat['CustomerID'] == 'A']
survival_coat = cph.predict_survival_function(customer_a_coat)

# Result: Similar shapes, different magnitudes
# (because product type is just another coefficient)
```

**Stratified:**
```python
# Predict for Customer A buying milk
customer_a_milk = df_milk[df_milk['CustomerID'] == 'A']
survival_milk = cph.predict_survival_function(customer_a_milk)
# Uses h₀_milk(t) - peaks at day 7

# Predict for Customer A buying coat
customer_a_coat = df_coat[df_coat['CustomerID'] == 'A']  
survival_coat = cph.predict_survival_function(customer_a_coat)
# Uses h₀_coat(t) - peaks at day 365

# Result: VERY different shapes reflecting different timing
```

---

## Summary Table

| Aspect | Non-Stratified | Stratified |
|--------|----------------|------------|
| **Baselines** | 1 for all | N (one per stratum) |
| **β coefficients** | Shared | Shared |
| **Can estimate stratum effect** | ✅ Yes | ❌ No |
| **Handles different timing** | ❌ Poor | ✅ Excellent |
| **Concordance (typical)** | 0.65-0.70 | 0.70-0.75 |
| **Predictions** | Less accurate | More accurate |
| **Use for diverse products** | ❌ Not recommended | ✅ Recommended |
| **Complexity** | Lower | Higher |
| **Data requirements** | Lower | Higher (need events per stratum) |

---

## Decision Tree: Which Model to Use?

```
Do you have groups with VERY different baseline timing?
(e.g., milk vs coats, young vs old patients)
    │
    ├─ YES → Do you have enough data per group (>20 events)?
    │         │
    │         ├─ YES → Use STRATIFIED ✅
    │         │
    │         └─ NO → Use group as COVARIATE instead
    │
    └─ NO → Are you violating proportional hazards?
              │
              ├─ YES → Consider STRATIFICATION
              │
              └─ NO → Use NON-STRATIFIED ✅
```

---

## Your Retail Repurchase Use Case

**Why Stratified is Perfect:**

✅ Products have VERY different repurchase cycles
   - Groceries: days to weeks
   - Personal care: weeks to months  
   - Durables: months to years

✅ Customer behaviors are consistent across products
   - Frequent shoppers buy ALL products more often
   - High spenders spend more on ALL products

✅ Enough data per product
   - Focused on popular products (30+ purchases)
   - Sufficient events to estimate baselines

✅ Focus is on customer targeting, not product comparison
   - Want to rank customers for each product
   - Don't need to compare milk vs coat directly

**Result:** Better predictions, better business decisions! 🎯
