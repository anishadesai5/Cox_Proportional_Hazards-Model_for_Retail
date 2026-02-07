"""
Stratified Cox Proportional Hazards Model for Customer Repurchase Prediction
Dataset: UCI Online Retail II (available on Kaggle)
Goal: Predict which customers are most likely to repurchase specific products and when
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Survival analysis
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from lifelines import KaplanMeierFitter

# ============================================================================
# STEP 1: DATA LOADING AND PREPARATION
# ============================================================================

def load_online_retail_data():
    """
    Load UCI Online Retail dataset from Kaggle or local file
    
    Dataset source: https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci
    Or: https://archive.ics.uci.edu/dataset/352/online+retail
    
    Columns:
    - InvoiceNo: Transaction ID (starts with 'C' if cancelled)
    - StockCode: Product/SKU ID  
    - Description: Product name
    - Quantity: Number of items purchased
    - InvoiceDate: Transaction timestamp
    - UnitPrice: Price per unit
    - CustomerID: Unique customer identifier
    - Country: Customer country
    """
    
    # Option 1: Load from Kaggle (after downloading)
    # Download from: https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci
    # Place in same directory as this script
    
    try:
        df = pd.read_excel('online_retail_II.xlsx')
        print("✅ Data loaded from local Excel file")
    except:
        try:
            df = pd.read_csv('online_retail_II.csv')
            print("✅ Data loaded from local CSV file")
        except:
            print("❌ Please download the dataset from Kaggle:")
            print("   https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci")
            print("   Save as 'online_retail_II.xlsx' or 'online_retail_II.csv'")
            return None
    
    return df


def clean_data(df):
    """Clean and prepare the dataset"""
    
    print(f"\n📊 Original dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # 1. Remove cancelled transactions (InvoiceNo starting with 'C')
    df = df[~df['Invoice'].astype(str).str.startswith('C')].copy()
    
    # 2. Remove rows with missing CustomerID
    df = df[df['Customer ID'].notna()].copy()
    
    # 3. Remove negative quantities and prices
    df = df[(df['Quantity'] > 0) & (df['Price'] > 0)].copy()
    
    # 4. Remove duplicates (EDA showed 3.22% duplicates)
    df = df.drop_duplicates()

    # 5. Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # 6. Create total revenue column
    df['Revenue'] = df['Quantity'] * df['Price']

    # 7. Ensure StockCode is string type
    df['StockCode'] = df['StockCode'].astype(str)

    # 8. Rename columns for convenience
    df = df.rename(columns={
        'Invoice': 'InvoiceNo',
        'StockCode': 'StockCode',
        'Customer ID': 'CustomerID',
        'InvoiceDate': 'InvoiceDate',
        'Price': 'UnitPrice'
    })
    
    print(f"✅ Cleaned dataset shape: {df.shape}")
    print(f"Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
    print(f"Unique customers: {df['CustomerID'].nunique()}")
    print(f"Unique products: {df['StockCode'].nunique()}")
    
    return df


# ============================================================================
# STEP 2: CREATE SURVIVAL ANALYSIS DATASET
# ============================================================================

def create_repurchase_dataset(df, min_purchases=2, min_product_purchases=30):
    """
    Transform transactional data into survival analysis format
    
    For each customer-product pair:
    - DURATION_DAYS: Days between purchases (or days since last purchase if censored)
    - EVENT: 1 if repurchased, 0 if censored (no repurchase observed)
    - Features: Customer behavior metrics
    
    Args:
        min_purchases: Minimum number of overall purchases for a customer to be included
        min_product_purchases: Minimum times a product must be purchased across all customers
    """
    
    print("\n🔄 Creating survival analysis dataset...")
    
    # Filter for products purchased frequently enough
    product_counts = df['StockCode'].value_counts()
    popular_products = product_counts[product_counts >= min_product_purchases].index
    df_filtered = df[df['StockCode'].isin(popular_products)].copy()
    
    print(f"Products with >={min_product_purchases} purchases: {len(popular_products)}")
    
    # Sort by customer and date
    df_filtered = df_filtered.sort_values(['CustomerID', 'StockCode', 'InvoiceDate'])
    
    # Create customer-product purchase history
    survival_records = []
    
    for (customer_id, stock_code), group in df_filtered.groupby(['CustomerID', 'StockCode']):
        
        # Skip if customer bought this product only once
        if len(group) < 2:
            continue
        
        # Get sorted purchase dates
        purchase_dates = group['InvoiceDate'].sort_values().values
        
        # Create records for consecutive purchases
        for i in range(len(purchase_dates) - 1):
            current_date = purchase_dates[i]
            next_date = purchase_dates[i + 1]
            duration_days = (next_date - current_date) / np.timedelta64(1, 'D')
            
            # Only include reasonable durations (1-365 days)
            if 1 <= duration_days <= 365:
                survival_records.append({
                    'CustomerID': customer_id,
                    'StockCode': stock_code,
                    'PurchaseDate': current_date,
                    'NextPurchaseDate': next_date,
                    'DURATION_DAYS': duration_days,
                    'EVENT': 1,  # Repurchase occurred
                    'Quantity': group.loc[group['InvoiceDate'] == current_date, 'Quantity'].sum(),
                    'Revenue': group.loc[group['InvoiceDate'] == current_date, 'Revenue'].sum(),
                })
    
    # Add censored observations (last purchase of each customer-product pair)
    observation_end = df_filtered['InvoiceDate'].max()
    
    for (customer_id, stock_code), group in df_filtered.groupby(['CustomerID', 'StockCode']):
        last_purchase = group['InvoiceDate'].max()
        duration_days = (observation_end - last_purchase) / np.timedelta64(1, 'D')
        
        if duration_days > 1:  # Only if there's meaningful censoring time
            survival_records.append({
                'CustomerID': customer_id,
                'StockCode': stock_code,
                'PurchaseDate': last_purchase,
                'NextPurchaseDate': None,
                'DURATION_DAYS': duration_days,
                'EVENT': 0,  # Censored (no repurchase observed)
                'Quantity': group.loc[group['InvoiceDate'] == last_purchase, 'Quantity'].sum(),
                'Revenue': group.loc[group['InvoiceDate'] == last_purchase, 'Revenue'].sum(),
            })
    
    survival_df = pd.DataFrame(survival_records)
    
    print(f"✅ Created {len(survival_df)} survival records")
    print(f"   Events (repurchases): {survival_df['EVENT'].sum()}")
    print(f"   Censored: {(1 - survival_df['EVENT']).sum()}")
    print(f"   Unique customers: {survival_df['CustomerID'].nunique()}")
    print(f"   Unique products: {survival_df['StockCode'].nunique()}")
    
    return survival_df


def engineer_features(survival_df, df_original):
    """
    Add customer-level features for Cox model
    
    Features to create:
    - RECENCY: Days since customer's first purchase in dataset
    - FREQUENCY: Total number of purchases by customer
    - MONETARY: Average spend per transaction
    - PRODUCT_FREQUENCY: Number of times customer bought this specific product
    - DAYS_SINCE_FIRST: Days since first purchase of this product
    """
    
    print("\n🔧 Engineering features...")
    
    # Customer-level aggregations
    customer_first_purchase = df_original.groupby('CustomerID')['InvoiceDate'].min()
    customer_purchase_count = df_original.groupby('CustomerID')['InvoiceNo'].nunique()
    customer_avg_revenue = df_original.groupby('CustomerID')['Revenue'].mean()
    
    # Customer-product aggregations
    customer_product_count = df_original.groupby(['CustomerID', 'StockCode'])['InvoiceNo'].nunique()
    customer_product_first = df_original.groupby(['CustomerID', 'StockCode'])['InvoiceDate'].min()
    
    # Add features to survival dataframe
    survival_df['RECENCY'] = survival_df.apply(
        lambda row: (row['PurchaseDate'] - customer_first_purchase[row['CustomerID']]).days,
        axis=1
    )
    
    survival_df['FREQUENCY'] = survival_df['CustomerID'].map(customer_purchase_count)
    survival_df['MONETARY'] = survival_df['CustomerID'].map(customer_avg_revenue)
    
    survival_df['PRODUCT_FREQUENCY'] = survival_df.apply(
        lambda row: customer_product_count.get((row['CustomerID'], row['StockCode']), 1),
        axis=1
    )
    
    survival_df['DAYS_SINCE_FIRST'] = survival_df.apply(
        lambda row: (row['PurchaseDate'] - customer_product_first[(row['CustomerID'], row['StockCode'])]).days,
        axis=1
    )
    
    # Log transform monetary to reduce skew
    survival_df['LOG_MONETARY'] = np.log1p(survival_df['MONETARY'])
    
    # Normalize features for better model convergence
    from sklearn.preprocessing import StandardScaler
    
    feature_cols = ['RECENCY', 'FREQUENCY', 'LOG_MONETARY', 'PRODUCT_FREQUENCY', 'DAYS_SINCE_FIRST']
    scaler = StandardScaler()
    survival_df[feature_cols] = scaler.fit_transform(survival_df[feature_cols])
    
    print(f"✅ Features engineered: {feature_cols}")
    print("\nFeature statistics:")
    print(survival_df[feature_cols].describe())
    
    return survival_df


# ============================================================================
# STEP 3: TRAIN STRATIFIED COX MODEL
# ============================================================================

def train_stratified_cox_model(survival_df, top_n_products=20):
    """
    Train stratified Cox Proportional Hazards model
    
    Stratification by StockCode means:
    - Each product gets its own baseline hazard h₀_product(t)
    - Customer features (β coefficients) are shared across all products
    
    Args:
        top_n_products: Use only top N most purchased products for cleaner results
    """
    
    print("\n🎯 Training Stratified Cox Proportional Hazards Model...")
    
    # Select top N products by number of purchases for demo
    top_products = survival_df.groupby('StockCode')['EVENT'].count().nlargest(top_n_products).index
    model_df = survival_df[survival_df['StockCode'].isin(top_products)].copy()
    
    print(f"Training on {len(model_df)} records from {len(top_products)} products")
    
    # Prepare features for model
    feature_cols = ['RECENCY', 'FREQUENCY', 'LOG_MONETARY', 'PRODUCT_FREQUENCY', 'DAYS_SINCE_FIRST']
    
    # Split into train/test (80/20)
    from sklearn.model_selection import train_test_split
    
    train_df, test_df = train_test_split(model_df, test_size=0.2, random_state=42)
    
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    # STRATIFIED Cox Model
    print("\n" + "="*60)
    print("TRAINING STRATIFIED COX MODEL")
    print("="*60)
    
    cph_stratified = CoxPHFitter(penalizer=0.01)  # Small L2 penalty for stability
    
    cph_stratified.fit(
        train_df[feature_cols + ['DURATION_DAYS', 'EVENT', 'StockCode']],
        duration_col='DURATION_DAYS',
        event_col='EVENT',
        strata=['StockCode'],  # 🔑 KEY: Separate baseline per product
        show_progress=True
    )
    
    print("\n📊 STRATIFIED MODEL SUMMARY:")
    cph_stratified.print_summary()
    
    # NON-STRATIFIED Cox Model for comparison
    print("\n" + "="*60)
    print("TRAINING NON-STRATIFIED COX MODEL (FOR COMPARISON)")
    print("="*60)
    
    cph_unstratified = CoxPHFitter(penalizer=0.01)
    
    cph_unstratified.fit(
        train_df[feature_cols + ['DURATION_DAYS', 'EVENT']],
        duration_col='DURATION_DAYS',
        event_col='EVENT',
        show_progress=True
    )
    
    print("\n📊 NON-STRATIFIED MODEL SUMMARY:")
    cph_unstratified.print_summary()
    
    # Evaluate both models
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    train_concordance_strat = cph_stratified.concordance_index_
    test_concordance_strat = cph_stratified.score(
        test_df[feature_cols + ['DURATION_DAYS', 'EVENT', 'StockCode']],
        scoring_method='concordance_index'
    )

    train_concordance_unstrat = cph_unstratified.concordance_index_
    test_concordance_unstrat = cph_unstratified.score(
        test_df[feature_cols + ['DURATION_DAYS', 'EVENT']],
        scoring_method='concordance_index'
    )
    
    print(f"\nSTRATIFIED MODEL:")
    print(f"  Train Concordance: {train_concordance_strat:.4f}")
    print(f"  Test Concordance:  {test_concordance_strat:.4f}")
    
    print(f"\nNON-STRATIFIED MODEL:")
    print(f"  Train Concordance: {train_concordance_unstrat:.4f}")
    print(f"  Test Concordance:  {test_concordance_unstrat:.4f}")
    
    print(f"\n🏆 Winner: {'STRATIFIED' if test_concordance_strat > test_concordance_unstrat else 'NON-STRATIFIED'}")
    print(f"Improvement: {(test_concordance_strat - test_concordance_unstrat) * 100:.2f}%")
    
    return cph_stratified, cph_unstratified, train_df, test_df, feature_cols


# ============================================================================
# STEP 4: PREDICT AND RANK CUSTOMERS FOR EACH PRODUCT
# ============================================================================

def predict_repurchase_risk(model, test_df, feature_cols, top_n_customers=10):
    """
    For each product, rank customers by repurchase likelihood

    Uses partial hazard (risk score) to rank customers
    Higher risk = More likely to repurchase soon
    """

    print("\n🎲 PREDICTING REPURCHASE RISK FOR EACH PRODUCT...")
    print("="*60)

    # Get baseline survival - handle tuple column names for stratified models
    baseline_survival = model.baseline_survival_
    print(f"\nBaseline survival columns type: {type(baseline_survival.columns[0])}")

    # Convert to dict for easier lookup
    if isinstance(baseline_survival.columns[0], tuple):
        baseline_dict = {col[0]: baseline_survival[col] for col in baseline_survival.columns}
        print("Note: Baseline columns are tuples, extracting first element")
    else:
        baseline_dict = {str(col): baseline_survival[col] for col in baseline_survival.columns}

    results = []

    # Get unique products in test set
    products = test_df['StockCode'].unique()[:5]  # Top 5 for demo

    for product in products:
        print(f"\n📦 Product: {product}")
        print("-" * 40)

        product_str = str(product)

        # Get customers who purchased this product
        product_df = test_df[test_df['StockCode'] == product_str].copy()

        if len(product_df) == 0:
            print("   No test data for this product")
            continue

        # Check if baseline exists for this product
        if product_str not in baseline_dict:
            print(f"   Warning: No baseline survival for product {product_str}")
            continue

        # Predict partial hazard (risk score) for each customer - exp(X*beta)
        partial_hazard = model.predict_partial_hazard(product_df[feature_cols])
        product_df['RISK_SCORE'] = partial_hazard.values

        # Get baseline survival for this product
        baseline_surv = baseline_dict[product_str]

        # Calculate survival probabilities: S(t|X) = S_0(t)^exp(X*beta)
        for horizon in [30, 60, 90]:
            valid_times = baseline_surv.index[baseline_surv.index <= horizon]
            closest_time = valid_times.max() if len(valid_times) > 0 else baseline_surv.index.min()

            base_surv_at_t = baseline_surv.loc[closest_time]
            survival_probs = base_surv_at_t ** partial_hazard.values
            product_df[f'PROB_{horizon}D'] = 1 - survival_probs

        # Sort by risk score (highest first)
        product_df = product_df.sort_values('RISK_SCORE', ascending=False)

        # Display top N customers
        print(f"\n🏆 Top {min(top_n_customers, len(product_df))} customers most likely to repurchase:")

        display_cols = ['CustomerID', 'RISK_SCORE', 'PROB_30D', 'PROB_60D', 'PROB_90D',
                       'FREQUENCY', 'PRODUCT_FREQUENCY', 'EVENT']

        available_cols = [c for c in display_cols if c in product_df.columns]
        print(product_df[available_cols].head(top_n_customers).to_string(index=False))

        # Store results
        for _, row in product_df.head(top_n_customers).iterrows():
            results.append({
                'StockCode': product,
                'CustomerID': row['CustomerID'],
                'RISK_SCORE': row['RISK_SCORE'],
                'PROB_30D': row.get('PROB_30D', np.nan),
                'PROB_60D': row.get('PROB_60D', np.nan),
                'PROB_90D': row.get('PROB_90D', np.nan),
                'Actual_Event': row['EVENT']
            })

    results_df = pd.DataFrame(results)

    return results_df


def visualize_predictions(model, test_df, feature_cols, product_code):
    """
    Visualize survival curves for a specific product
    """

    print(f"\n📈 VISUALIZING PREDICTIONS FOR PRODUCT: {product_code}")
    print("="*60)

    product_str = str(product_code)
    product_df = test_df[test_df['StockCode'] == product_str].copy()

    if len(product_df) == 0:
        print(f"No test data for product {product_code}")
        return

    # Get baseline survival for this product
    baseline_survival = model.baseline_survival_
    if isinstance(baseline_survival.columns[0], tuple):
        baseline_dict = {col[0]: baseline_survival[col] for col in baseline_survival.columns}
    else:
        baseline_dict = {str(col): baseline_survival[col] for col in baseline_survival.columns}

    if product_str not in baseline_dict:
        print(f"No baseline survival for product {product_code}")
        return

    baseline_surv = baseline_dict[product_str]

    # Get top 5 highest and lowest risk customers
    partial_hazard = model.predict_partial_hazard(product_df[feature_cols])
    product_df['RISK_SCORE'] = partial_hazard.values

    top_5 = product_df.nlargest(5, 'RISK_SCORE')
    bottom_5 = product_df.nsmallest(5, 'RISK_SCORE')

    # Compute survival functions: S(t|X) = S_0(t)^exp(X*beta)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: High risk customers
    ax1 = axes[0]
    top_5_hazards = model.predict_partial_hazard(top_5[feature_cols])
    for i, (idx, row) in enumerate(top_5.iterrows()):
        survival_curve = baseline_surv ** top_5_hazards.iloc[i]
        ax1.plot(survival_curve.index, survival_curve.values, label=f"Customer {int(row['CustomerID'])}")

    ax1.set_title(f'Product {product_code}: HIGH RISK Customers\n(Most likely to repurchase)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Days Since Last Purchase', fontsize=12)
    ax1.set_ylabel('Survival Probability (No Repurchase)', fontsize=12)
    ax1.legend(title='Top 5 Risk Customers')
    ax1.grid(alpha=0.3)

    # Plot 2: Low risk customers
    ax2 = axes[1]
    bottom_5_hazards = model.predict_partial_hazard(bottom_5[feature_cols])
    for i, (idx, row) in enumerate(bottom_5.iterrows()):
        survival_curve = baseline_surv ** bottom_5_hazards.iloc[i]
        ax2.plot(survival_curve.index, survival_curve.values, label=f"Customer {int(row['CustomerID'])}")

    ax2.set_title(f'Product {product_code}: LOW RISK Customers\n(Least likely to repurchase)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Days Since Last Purchase', fontsize=12)
    ax2.set_ylabel('Survival Probability (No Repurchase)', fontsize=12)
    ax2.legend(title='Bottom 5 Risk Customers')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'survival_curves_product_{product_code}.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved survival curves to survival_curves_product_{product_code}.png")

    return fig


# ============================================================================
# STEP 5: BUSINESS INSIGHTS
# ============================================================================

def generate_business_insights(results_df, model, feature_cols):
    """
    Generate actionable business recommendations
    """
    
    print("\n💼 BUSINESS INSIGHTS & RECOMMENDATIONS")
    print("="*60)
    
    # 1. Feature importance (coefficient magnitude)
    coef_df = model.summary[['coef', 'exp(coef)', 'p']].copy()
    coef_df['abs_coef'] = coef_df['coef'].abs()
    coef_df = coef_df.sort_values('abs_coef', ascending=False)
    
    print("\n1️⃣ MOST IMPORTANT FACTORS FOR REPURCHASE:")
    print(coef_df)
    
    print("\n📝 Interpretation:")
    for feature in coef_df.index[:3]:
        coef = coef_df.loc[feature, 'coef']
        hr = coef_df.loc[feature, 'exp(coef)']
        p_val = coef_df.loc[feature, 'p']
        
        if p_val < 0.05:
            direction = "INCREASES" if coef > 0 else "DECREASES"
            print(f"\n  • {feature}:")
            print(f"    - {direction} repurchase risk by {abs((hr - 1) * 100):.1f}% per unit increase")
            print(f"    - Hazard Ratio: {hr:.3f}")
            print(f"    - p-value: {p_val:.4f} ✓ (significant)")
    
    # 2. Product-level insights
    print("\n2️⃣ PRODUCT-LEVEL TARGETING OPPORTUNITIES:")
    
    product_summary = results_df.groupby('StockCode').agg({
        'PROB_30D': 'mean',
        'PROB_60D': 'mean',
        'PROB_90D': 'mean',
        'RISK_SCORE': 'mean',
        'CustomerID': 'count'
    }).sort_values('PROB_30D', ascending=False)
    
    print("\nProducts with highest average 30-day repurchase probability:")
    print(product_summary.head())
    
    # 3. Customer segmentation
    print("\n3️⃣ CUSTOMER SEGMENTATION FOR TARGETING:")
    
    # High-intent customers (>60% prob in 30 days)
    high_intent = results_df[results_df['PROB_30D'] > 0.6]
    
    # Medium-intent customers (30-60% prob in 30 days)
    medium_intent = results_df[(results_df['PROB_30D'] > 0.3) & (results_df['PROB_30D'] <= 0.6)]
    
    # Low-intent customers (<30% prob in 30 days)
    low_intent = results_df[results_df['PROB_30D'] <= 0.3]
    
    print(f"\n  • HIGH INTENT (>60% prob in 30 days): {len(high_intent)} customers")
    print(f"    → Strategy: Gentle reminder email or push notification")
    print(f"    → They're already primed to buy - don't over-incentivize")
    
    print(f"\n  • MEDIUM INTENT (30-60% prob in 30 days): {len(medium_intent)} customers")
    print(f"    → Strategy: Offer modest discount (10-15%) to accelerate purchase")
    print(f"    → Personalized product recommendations")
    
    print(f"\n  • LOW INTENT (<30% prob in 30 days): {len(low_intent)} customers")
    print(f"    → Strategy: Skip for now or try re-engagement campaign")
    print(f"    → Avoid annoying them with irrelevant offers")
    
    # 4. Expected demand forecast
    print("\n4️⃣ INVENTORY & DEMAND FORECASTING:")
    
    for horizon in [30, 60, 90]:
        col = f'PROB_{horizon}D'
        if col in results_df.columns:
            expected_purchases = results_df[col].sum()
            print(f"\n  • Expected repurchases in next {horizon} days: {expected_purchases:.0f}")
            print(f"    → Use for inventory planning and demand forecasting")
    
    print("\n" + "="*60)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Run complete pipeline
    """
    
    print("\n" + "="*60)
    print("STRATIFIED COX MODEL FOR CUSTOMER REPURCHASE PREDICTION")
    print("="*60)
    
    # Step 1: Load and clean data
    df = load_online_retail_data()
    if df is None:
        return
    
    df_clean = clean_data(df)
    
    # Step 2: Create survival dataset
    survival_df = create_repurchase_dataset(df_clean, min_purchases=2, min_product_purchases=30)
    survival_df = engineer_features(survival_df, df_clean)
    
    # Step 3: Train models
    cph_stratified, cph_unstratified, train_df, test_df, feature_cols = train_stratified_cox_model(survival_df)
    
    # Step 4: Predictions and rankings
    results_df = predict_repurchase_risk(cph_stratified, test_df, feature_cols)
    
    # Step 5: Visualizations
    # Pick a product with good test data
    if len(test_df['StockCode'].unique()) > 0:
        sample_product = test_df['StockCode'].value_counts().index[0]
        visualize_predictions(cph_stratified, test_df, feature_cols, sample_product)
    
    # Step 6: Business insights
    generate_business_insights(results_df, cph_stratified, feature_cols)
    
    # Save results
    results_df.to_csv('repurchase_predictions.csv', index=False)
    print(f"\n✅ Results saved to repurchase_predictions.csv")
    
    return cph_stratified, results_df


if __name__ == "__main__":
    model, results = main()
