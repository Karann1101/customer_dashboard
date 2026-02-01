# app.py -- Streamlit dashboard with Customer Satisfaction + RFM
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

st.set_page_config(page_title="Customer Satisfaction + RFM Dashboard", layout="wide")
sns.set_style("whitegrid")

# ---------------------------
# Helpers: build analytics from raw files (if needed)
# ---------------------------
BASE_RAW_FOLDER_NAMES = [
    "brazilian-ecommerce",
    "brazillian-ecommerce",
    "brazilian-ecommerce-dataset",
]



RAW_FILES = {
    "orders": "olist_orders_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "category_translation": "product_category_name_translation.csv"
}

def find_raw_folder():
    # search current dir and parent for likely folders
    for cand in ["."] + BASE_RAW_FOLDER_NAMES:
        for root in [cand, os.path.join(".", cand), os.path.join("..", cand)]:
            if os.path.isdir(root):
                # check if some expected files exist inside
                if any(os.path.exists(os.path.join(root, f)) for f in RAW_FILES.values()):
                    return root
    return None

@st.cache_data
def build_analytics_from_raw():
    """
    Attempt to construct an order-level analytics dataframe from raw Olist files.
    Returns dataframe or None if required files not found.
    """
    raw_root = find_raw_folder()
    if raw_root is None:
        return None

    try:
        orders = pd.read_csv(os.path.join(raw_root, RAW_FILES["orders"]),
                             parse_dates=["order_purchase_timestamp",
                                          "order_approved_at",
                                          "order_delivered_carrier_date",
                                          "order_delivered_customer_date",
                                          "order_estimated_delivery_date"],
                             low_memory=False)
        customers = pd.read_csv(os.path.join(raw_root, RAW_FILES["customers"]), low_memory=False)
        order_items = pd.read_csv(os.path.join(raw_root, RAW_FILES["order_items"]), low_memory=False)
        products = pd.read_csv(os.path.join(raw_root, RAW_FILES["products"]), low_memory=False)
        payments = pd.read_csv(os.path.join(raw_root, RAW_FILES["payments"]), low_memory=False)
        reviews = pd.read_csv(os.path.join(raw_root, RAW_FILES["reviews"]),
                              parse_dates=["review_creation_date", "review_answer_timestamp"],
                              low_memory=False)
        cat_trans = None
        ct_path = os.path.join(raw_root, RAW_FILES["category_translation"])
        if os.path.exists(ct_path):
            cat_trans = pd.read_csv(ct_path, low_memory=False)

        # aggregate order items to order-level price & freight & items_count
        oi_agg = (
            order_items
            .groupby("order_id")
            .agg(
                price=("price", "sum"),
                freight_value=("freight_value", "sum"),
                items_count=("order_item_id", "count")
            )
            .reset_index()
        )

        # choose dominant product category per order (first)
        order_product = (
            order_items.merge(products, on="product_id", how="left")
            .merge(cat_trans, on="product_category_name", how="left") if cat_trans is not None
            else order_items.merge(products, on="product_id", how="left")
        )
        # prefer translated name if available
        if cat_trans is not None and "product_category_name_english" in order_product.columns:
            prod_cat = order_product.groupby("order_id")["product_category_name_english"].first().reset_index()
            prod_cat.columns = ["order_id", "product_category"]
        else:
            prod_cat = order_product.groupby("order_id")["product_category_name"].first().reset_index()
            prod_cat.columns = ["order_id", "product_category"]

        # aggregate payments
        payments_agg = (
            payments
            .groupby("order_id")
            .agg(
                payment_type=("payment_type", lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
                payment_installments=("payment_installments", "max"),
                payment_value_sum=("payment_value", "sum")
            )
            .reset_index()
        )

        # merge everything
        df = orders.merge(customers, on="customer_id", how="left")
        df = df.merge(reviews[["order_id", "review_score"]], on="order_id", how="left")
        df = df.merge(oi_agg, on="order_id", how="left")
        df = df.merge(prod_cat, on="order_id", how="left")
        df = df.merge(payments_agg, on="order_id", how="left")

        # normalize column names expected by dashboard
        if "customer_unique_id" in customers.columns:
            df["customer_unique_id"] = df.get("customer_unique_id", df["customer_id"])
        else:
            df["customer_unique_id"] = df["customer_id"]

        # Save for reuse
        try:
            df.to_csv("analytics_data.csv", index=False)
        except Exception:
            pass

        return df

    except Exception as e:
        # if anything fails, return None
        return None


@st.cache_data
def load_data():
    # Try analytics_data.csv first
    if os.path.exists("analytics_data.csv"):
        try:
            df = pd.read_csv("analytics_data.csv", low_memory=False, parse_dates=[
                "order_purchase_timestamp",
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
                "review_creation_date",
                "review_answer_timestamp"
            ])
            return df
        except Exception:
            # if parsing fails, try without parse_dates
            df = pd.read_csv("analytics_data.csv", low_memory=False)
            return df

    # Else try to build from raw files
    df = build_analytics_from_raw()
    return df

# ---------------------------
# Load / prepare dataset
# ---------------------------
df = load_data()

if df is None:
    st.error(
        "Could not find analytics_data.csv or the raw 'brazilian-ecommerce' files in the working directory. "
        "Please place 'analytics_data.csv' in this folder OR put the Olist CSV files in a folder named "
        "'brazilian-ecommerce' (or one of similar names) alongside this app."
    )
    st.stop()

# Ensure column names we use exist or create fallbacks
if "customer_unique_id" not in df.columns:
    if "customer_id" in df.columns:
        df["customer_unique_id"] = df["customer_id"]
    else:
        st.error("Dataset lacks 'customer_id' or 'customer_unique_id'. Cannot compute customer-level metrics.")
        st.stop()

# ---------------------------
# Ensure satisfaction features exist (create if missing)
# ---------------------------
# review_score
if "review_score" not in df.columns and "review_id" in df.columns:
    st.warning("No review_score column in analytics_data.csv; review info may be missing.")
    df["review_score"] = np.nan

df["is_unhappy"] = ((df.get("review_score", pd.Series(np.nan)) <= 2)).astype(int)

# delivery delay
if "order_delivered_customer_date" in df.columns and "order_estimated_delivery_date" in df.columns:
    if not pd.api.types.is_datetime64_any_dtype(df["order_delivered_customer_date"]):
        df["order_delivered_customer_date"] = pd.to_datetime(df["order_delivered_customer_date"], errors="coerce")
    if not pd.api.types.is_datetime64_any_dtype(df["order_estimated_delivery_date"]):
        df["order_estimated_delivery_date"] = pd.to_datetime(df["order_estimated_delivery_date"], errors="coerce")

    df["delivery_delay_days"] = (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]).dt.days
else:
    df["delivery_delay_days"] = np.nan

def delay_bucket(x):
    if pd.isna(x):
        return "Unknown"
    elif x <= 0:
        return "On Time"
    elif x <= 3:
        return "Late (1–3 days)"
    else:
        return "Very Late (>3 days)"

df["delay_bucket"] = df["delivery_delay_days"].apply(delay_bucket)

# freight-to-price ratio
if "freight_value" in df.columns and "price" in df.columns:
    df["freight_to_price_ratio"] = df["freight_value"] / (df["price"].replace(0, np.nan))
    # cap outliers for visualization
    df.loc[df["freight_to_price_ratio"] > 3, "freight_to_price_ratio"] = 3
else:
    df["freight_to_price_ratio"] = np.nan

# ---------------------------
# Compute RFM at customer level
# ---------------------------
def compute_rfm(orders_df, customer_id_col="customer_unique_id", date_col="order_purchase_timestamp", monetary_col="payment_value_sum"):
    # orders_df: order-level dataframe
    if date_col not in orders_df.columns:
        st.warning("Orders data does not have purchase timestamps. RFM Recency cannot be computed.")
        return None

    # ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(orders_df[date_col]):
        orders_df[date_col] = pd.to_datetime(orders_df[date_col], errors="coerce")

    # Monetary fallback: prefer payment_value_sum, else price
    if monetary_col not in orders_df.columns:
        if "price" in orders_df.columns:
            orders_df["monetary_value"] = orders_df["price"]
        else:
            orders_df["monetary_value"] = 0.0
    else:
        orders_df["monetary_value"] = orders_df.get(monetary_col, 0.0)

    reference_date = orders_df[date_col].max() + timedelta(days=1)

    rfm = orders_df.groupby(customer_id_col).agg(
        recency_days = (date_col, lambda x: (reference_date - x.max()).days),
        frequency = (date_col, "count"),
        monetary = ("monetary_value", "sum")
    ).reset_index()

    # Remove zero-monetary customers (if any)
    rfm = rfm[rfm["monetary"] > 0]

    # Score each metric 1 - 5 (quintiles). For recency smaller is better -> invert.
    try:
        rfm["r_score"] = pd.qcut(rfm["recency_days"], 5, labels=[5,4,3,2,1]).astype(int)
    except Exception:
        # fallback: rank-based
        rfm["r_score"] = pd.cut(rfm["recency_days"], bins=5, labels=[5,4,3,2,1]).astype(int)

    try:
        rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
    except Exception:
        rfm["f_score"] = pd.cut(rfm["frequency"], bins=5, labels=[1,2,3,4,5]).astype(int)

    try:
        rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
    except Exception:
        rfm["m_score"] = pd.cut(rfm["monetary"], bins=5, labels=[1,2,3,4,5]).astype(int)

    rfm["rfm_score"] = rfm["r_score"].astype(str) + rfm["f_score"].astype(str) + rfm["m_score"].astype(str)
    rfm["rfm_sum"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]

    # simple segment mapping by rfm_sum
    def label_segment(s):
        if s >= 13:
            return "Champions"
        elif s >= 10:
            return "Loyal"
        elif s >= 8:
            return "Potential Loyalists"
        elif s >= 6:
            return "Need Attention"
        elif s >= 4:
            return "At Risk"
        else:
            return "Lost"

    rfm["rfm_segment"] = rfm["rfm_sum"].apply(label_segment)
    return rfm

rfm_table = compute_rfm(df)

# ---------------------------
# SIDEBAR: filters and segment selection
# ---------------------------
st.sidebar.markdown("## 🔎 Filters")
st.sidebar.caption("Use filters to explore specific customer segments")

# reuse or redefine filters with safety if columns missing
state_options = sorted(df["customer_state"].dropna().unique()) if "customer_state" in df.columns else []
category_options = sorted(df["product_category"].dropna().unique()) if "product_category" in df.columns else []

state_filter = st.sidebar.multiselect("Customer State", options=state_options, default=state_options)
category_filter = st.sidebar.multiselect("Product Category", options=category_options, default=category_options)

segment_options = sorted(rfm_table["rfm_segment"].unique()) if rfm_table is not None else []
segment_filter = st.sidebar.multiselect("RFM Segment", options=segment_options, default=segment_options)

# apply filters to order-level df
mask = pd.Series(True, index=df.index)
if state_filter:
    if "customer_state" in df.columns:
        mask &= df["customer_state"].isin(state_filter)
if category_filter:
    if "product_category" in df.columns:
        mask &= df["product_category"].isin(category_filter)
df_filtered = df[mask].copy()

# filter rfm_table by segment and also by customers present in df_filtered
if rfm_table is not None:
    # keep only customers who appear in filtered orders
    eligible_customers = df_filtered["customer_unique_id"].unique()
    rfm_filtered = rfm_table[rfm_table["customer_unique_id"].isin(eligible_customers)]
    if segment_filter:
        rfm_filtered = rfm_filtered[rfm_filtered["rfm_segment"].isin(segment_filter)]
else:
    rfm_filtered = None

# ---------------------------
# PAGE LAYOUT
# ---------------------------
st.markdown(
    """
    <h1 style="margin-bottom:0">📊 Customer Intelligence Dashboard</h1>
    <p style="color:gray; margin-top:4px">
    RFM-based customer segmentation & satisfaction insights
    </p>
    """,
    unsafe_allow_html=True
)

# Executive KPIs (order-level / filtered)
col1, col2, col3, col4 = st.columns(4)
col1.metric("⭐ Average Rating", round(df_filtered["review_score"].mean(), 2))
col2.metric("😟 Unhappy Customers", f"{round(df_filtered['is_unhappy'].mean()*100,2)}%")
col3.metric("🚚 Late Deliveries", f"{round((df_filtered['delivery_delay_days']>0).mean()*100,2)}%")
col4.metric("💸 Shipping Cost Ratio", round(df_filtered["freight_to_price_ratio"].mean(), 2))

st.markdown("---")

# Delivery impact (satisfaction)
st.subheader("🚚 Delivery Experience Impact on Customer Satisfaction")
if "delay_bucket" in df_filtered.columns:
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        df_filtered.groupby("delay_bucket")["review_score"].mean().plot(kind="bar", ax=ax)
        ax.set_ylabel("Average Review Score")
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots()
        (df_filtered.groupby("delay_bucket")["is_unhappy"].mean()*100).plot(kind="bar", ax=ax, color="red")
        ax.set_ylabel("Unhappy Customers (%)")
        st.pyplot(fig)
else:
    st.warning("delay_bucket not available in dataset.")

st.markdown("---")

# Freight cost impact
st.subheader("💸 Freight Cost Impact")
if "freight_to_price_ratio" in df_filtered.columns:
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_filtered, x="freight_to_price_ratio", y="review_score", alpha=0.3, ax=ax)
    ax.set_xlabel("Freight / Price Ratio")
    ax.set_ylabel("Review Score")
    st.pyplot(fig)
else:
    st.warning("freight_to_price_ratio not available in dataset.")

st.markdown("---")

# Payment experience
st.subheader("💳 Payment Method vs Satisfaction")
if "payment_type" in df_filtered.columns:
    fig, ax = plt.subplots()
    df_filtered.groupby("payment_type")["review_score"].mean().sort_values().plot(kind="barh", ax=ax)
    ax.set_xlabel("Average Review Score")
    st.pyplot(fig)
else:
    st.warning("payment_type column not present; payment analysis unavailable.")

st.markdown("---")

# Product category risk table
st.subheader("📦 Product Category Satisfaction Risk")
if "product_category" in df_filtered.columns:
    category_stats = (
        df_filtered.groupby("product_category")
        .agg(avg_review=("review_score", "mean"),
             unhappy_rate=("is_unhappy", "mean"),
             orders=("review_score", "count"))
        .query("orders > 50")
        .sort_values("unhappy_rate", ascending=False)
    )
    st.dataframe(
    category_stats
    .assign(unhappy_rate=lambda x: (x["unhappy_rate"]*100).round(2)),
    use_container_width=True
    )

else:
    st.warning("product_category not present.")

st.markdown("---")

# Geographic satisfaction
st.subheader("📍 Satisfaction by State")
if "customer_state" in df_filtered.columns:
    fig, ax = plt.subplots(figsize=(10,4))
    df_filtered.groupby("customer_state")["review_score"].mean().sort_values().plot(kind="bar", ax=ax)
    ax.set_ylabel("Average Review Score")
    st.pyplot(fig)
else:
    st.warning("customer_state not present.")

st.markdown("---")

# ---------------------------
# RFM Section
# ---------------------------
st.subheader("🏷 Customer Segmentation (RFM Analysis)")
st.caption(
    "Customers grouped by Recency, Frequency, and Monetary value to guide retention and growth strategies."
)
if rfm_filtered is None:
    st.warning("RFM table could not be computed (missing order timestamps or monetary info).")
else:
    # RFM KPIs
    seg_counts = rfm_filtered["rfm_segment"].value_counts().rename_axis("segment").reset_index(name="customers")
    seg_revenue = (rfm_filtered.merge(df_filtered[["customer_unique_id", "monetary_value"]].groupby("customer_unique_id").sum().reset_index(),
                                      on="customer_unique_id", how="left")
                   .groupby("rfm_segment")["monetary_value"].sum().reset_index(name="revenue"))
    seg_summary = seg_counts.merge(seg_revenue, left_on="segment", right_on="rfm_segment", how="left").fillna(0)
    seg_summary = seg_summary[["segment","customers","revenue"]].sort_values("revenue", ascending=False)

    c1, c2 = st.columns([1,2])
    with c1:
        st.write("Customers by Segment")
        st.bar_chart(seg_counts.set_index("segment")["customers"])
    with c2:
        st.write("Revenue by Segment")
        st.bar_chart(seg_summary.set_index("segment")["revenue"])

    st.markdown("**Top customers (by monetary)**")
    top_customers = rfm_filtered.sort_values("monetary", ascending=False).head(20)
    st.dataframe(top_customers[["customer_unique_id","recency_days","frequency","monetary","rfm_segment","rfm_score","rfm_sum"]])

    st.markdown("**RFM scatter (Frequency vs Monetary), colored by Recency (days)**")
    fig, ax = plt.subplots(figsize=(8,5))
    sc = ax.scatter(rfm_filtered["frequency"], rfm_filtered["monetary"], c=rfm_filtered["recency_days"], cmap="viridis", alpha=0.7)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Monetary (Total)")
    plt.colorbar(sc, label="Recency (days)")
    st.pyplot(fig)

st.markdown("---")

# Insights block (editable)
st.subheader("🧠 Key Business Insights & Actions")
st.caption("Actionable recommendations derived from the data:")
st.markdown("""
- **RFM**: Identify 'Champions' for loyalty programs and 'At Risk' for win-back campaigns.  
- **Delivery**: Prioritize logistics improvement for states/categories with high unhappy rates.  
- **Freight**: Consider free-shipping thresholds or bundling for orders with high freight/price ratios.  
- **Payment**: Make checkout smoother for payment types correlated with lower reviews.  
""")


# End of app
