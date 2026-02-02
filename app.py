# app.py -- Streamlit dashboard with Customer Satisfaction + RFM
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
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

BRAZIL_STATE_MAP = {
    "AC": "Acre",
    "AL": "Alagoas",
    "AP": "Amapá",
    "AM": "Amazonas",
    "BA": "Bahia",
    "CE": "Ceará",
    "DF": "Distrito Federal",
    "ES": "Espírito Santo",
    "GO": "Goiás",
    "MA": "Maranhão",
    "MT": "Mato Grosso",
    "MS": "Mato Grosso do Sul",
    "MG": "Minas Gerais",
    "PA": "Pará",
    "PB": "Paraíba",
    "PR": "Paraná",
    "PE": "Pernambuco",
    "PI": "Piauí",
    "RJ": "Rio de Janeiro",
    "RN": "Rio Grande do Norte",
    "RS": "Rio Grande do Sul",
    "RO": "Rondônia",
    "RR": "Roraima",
    "SC": "Santa Catarina",
    "SP": "São Paulo",
    "SE": "Sergipe",
    "TO": "Tocantins"
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

if "customer_state" in df.columns:
    df["customer_state_full"] = df["customer_state"].map(BRAZIL_STATE_MAP)


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
category_options = sorted(df["product_category"].dropna().unique()) if "product_category" in df.columns else []
state_options = (
    sorted(df["customer_state_full"].dropna().unique())
    if "customer_state_full" in df.columns
    else []
)

category_filter = st.sidebar.multiselect("Product Category", options=category_options, default=category_options)
state_filter = st.sidebar.multiselect(
    "Customer State",
    options=state_options,
    default=state_options
)


segment_options = sorted(rfm_table["rfm_segment"].unique()) if rfm_table is not None else []
segment_filter = st.sidebar.multiselect("RFM Segment", options=segment_options, default=segment_options)

# apply filters to order-level df
mask = pd.Series(True, index=df.index)
if state_filter:
    if "customer_state" in df.columns:
        mask &= df["customer_state_full"].isin(state_filter)
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
# PAGE LAYOUT (interactive, tabbed)
# ---------------------------
# This whole block replaces previous page-layout sections and adds interactivity:
tab_overview, tab_rfm, tab_ops = st.tabs([
    "📊 Overview",
    "🏷 RFM & Segments",
    "🚚 Ops & Satisfaction"
])

# Prepare a revenue fallback (use RFM monetary if present, else payments/price)
if rfm_table is not None and "monetary" in rfm_table.columns:
    total_revenue = rfm_table["monetary"].sum()
elif "payment_value" in df.columns:
    total_revenue = df["payment_value"].sum()
elif "price" in df.columns:
    total_revenue = df["price"].sum()
else:
    total_revenue = 0.0

# ========== TAB 1: Overview ==========
with tab_overview:
    # Header
    st.markdown(
        """
        <h2 style="margin:0">📊 Executive Overview</h2>
        <p style="color:gray; margin-top:4px">High-level KPIs and quick filters</p>
        """,
        unsafe_allow_html=True
    )

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("👥 Customers (unique)", df["customer_unique_id"].nunique())
    k2.metric("💰 Estimated Revenue", f"₹ {total_revenue:,.0f}")
    k3.metric("⭐ Avg Rating", round(df_filtered["review_score"].mean(), 2) if "review_score" in df_filtered.columns else "N/A")
    k4.metric("⚠️ Unhappy %", f"{round(df_filtered['is_unhappy'].mean()*100,2)}%" if "is_unhappy" in df_filtered.columns else "N/A")

    st.markdown("---")

    # Small summary row
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Trends & quick views")
        # trend placeholder: orders over time if available
        if "order_purchase_timestamp" in df.columns:
            orders_ts = df_filtered.set_index("order_purchase_timestamp").resample("W").size()
            fig, ax = plt.subplots()
            orders_ts.plot(ax=ax)
            ax.set_title("Orders per week (filtered)")
            ax.set_ylabel("Orders")
            st.pyplot(fig)
        else:
            st.info("No order timestamp available for trend plots.")
    with c2:
        st.subheader("Quick Actions")
        st.write("- Target `At Risk` with email coupon")
        st.write("- Investigate top categories with high unhappy rates")
        st.write("- Inspect sellers in low-rating states")

# ========== TAB 2: RFM & Segments ==========
with tab_rfm:
    st.markdown(
        "<h2 style='margin:0'>🏷 RFM & Segment Analysis</h2>",
        unsafe_allow_html=True
    )
    st.caption("Compare segments, explore top customers, and visualize customer value.")

    if rfm_table is None:
        st.warning("RFM could not be computed — missing timestamps or monetary info.")
    else:
        # Segment comparison controls
        st.markdown("### 🔍 Compare segments")
        seg_options = sorted(rfm_filtered["rfm_segment"].unique()) if (rfm_filtered is not None and not rfm_filtered.empty) else sorted(rfm_table["rfm_segment"].unique())
        left, right = st.columns(2)
        with left:
            seg_a = st.selectbox("Segment A", options=seg_options, index=0)
        with right:
            seg_b = st.selectbox("Segment B", options=seg_options, index=1 if len(seg_options)>1 else 0)

        seg_a_df = rfm_table[rfm_table["rfm_segment"] == seg_a]
        seg_b_df = rfm_table[rfm_table["rfm_segment"] == seg_b]

        ca, cb = st.columns(2)
        ca.metric(f"{seg_a} — Customers", seg_a_df.shape[0])
        ca.metric(f"{seg_a} — Avg Spend", f"₹ {seg_a_df['monetary'].mean():,.0f}")
        cb.metric(f"{seg_b} — Customers", seg_b_df.shape[0])
        cb.metric(f"{seg_b} — Avg Spend", f"₹ {seg_b_df['monetary'].mean():,.0f}")

        st.markdown("---")
        # Revenue by segment (bar)
        seg_revenue = rfm_table.groupby("rfm_segment")["monetary"].sum().sort_values(ascending=False)
        fig, ax = plt.subplots()
        seg_revenue.plot(kind="bar", ax=ax)
        ax.set_title("Revenue by RFM Segment")
        ax.set_ylabel("Revenue")
        st.pyplot(fig)

        st.markdown("---")
        # interactive scatter with Plotly
        st.markdown("### 💎 Customer Value Distribution")
        fig_px = px.scatter(
            rfm_table,
            x="frequency",
            y="monetary",
            color="rfm_segment",
            hover_data=["recency_days"],
            title="Frequency vs Monetary (hover for Recency)",
            width=900,
            height=450
        )
        st.plotly_chart(fig_px, use_container_width=True)

        # Top customers expander
        with st.expander("🏆 Top 20 customers (by monetary)"):
            top_customers = rfm_table.sort_values("monetary", ascending=False).head(20)
            st.dataframe(top_customers[["customer_unique_id","recency_days","frequency","monetary","rfm_score","rfm_segment"]], use_container_width=True)

        # At-risk expander
        with st.expander("⚠️ At Risk Customers (sample)"):
            at_risk = rfm_table[rfm_table["rfm_segment"] == "At Risk"].sort_values("recency_days", ascending=False).head(50)
            st.dataframe(at_risk[["customer_unique_id","recency_days","frequency","monetary"]], use_container_width=True)

# ========== TAB 3: Operations & Satisfaction ==========
with tab_ops:
    st.markdown("<h2 style='margin:0'>🚚 Operations & Customer Satisfaction</h2>", unsafe_allow_html=True)
    st.caption("Drill into delivery, freight and payment impacts on satisfaction.")

    # interactive metric choice for delivery charts
    metric_choice = st.radio(
        "Metric to analyze by delivery performance:",
        options=["review_score", "is_unhappy", "freight_to_price_ratio"],
        horizontal=True
    )

    # slider for late threshold
    late_threshold = st.slider(
        "Late delivery threshold (days after estimated delivery)",
        min_value=0,
        max_value=10,
        value=3
    )

    # compute custom late flag
    if "delivery_delay_days" in df_filtered.columns:
        df_filtered["is_late_custom"] = (df_filtered["delivery_delay_days"] > late_threshold).astype(int)
    else:
        df_filtered["is_late_custom"] = 0

    st.markdown("### Delivery performance impact")
    c1, c2 = st.columns(2)
    with c1:
        # show chosen metric aggregated by delay_bucket
        if "delay_bucket" in df_filtered.columns:
            agg = df_filtered.groupby("delay_bucket")[metric_choice].mean()
            fig, ax = plt.subplots()
            agg.plot(kind="bar", ax=ax)
            ax.set_ylabel(metric_choice.replace("_"," ").title())
            ax.set_title(f"{metric_choice.replace('_',' ').title()} by Delivery Bucket")
            st.pyplot(fig)
        else:
            st.info("No delivery_bucket data available.")

    with c2:
        if "delivery_delay_days" in df_filtered.columns:
            st.metric("Late Delivery Rate (custom)", f"{df_filtered['is_late_custom'].mean()*100:.2f}%")
            # show distribution of delay
            fig, ax = plt.subplots()
            df_filtered["delivery_delay_days"].dropna().hist(bins=30, ax=ax)
            ax.set_title("Delivery Delay Distribution (days)")
            st.pyplot(fig)
        else:
            st.info("No delivery delay data available.")

    st.markdown("---")
    # Freight cost interactive
    st.markdown("### 💸 Freight vs Satisfaction")
    if "freight_to_price_ratio" in df_filtered.columns and "review_score" in df_filtered.columns:
        scatter_choice = st.checkbox("Color by RFM segment", value=True)
        fig, ax = plt.subplots()
        if scatter_choice and rfm_table is not None:
            # merge small sample to enrich
            merged = df_filtered.merge(rfm_table[["customer_unique_id","rfm_segment"]], on="customer_unique_id", how="left")
            sns.scatterplot(data=merged, x="freight_to_price_ratio", y="review_score", hue="rfm_segment", alpha=0.4, ax=ax)
        else:
            sns.scatterplot(data=df_filtered, x="freight_to_price_ratio", y="review_score", alpha=0.4, ax=ax)
        ax.set_xlabel("Freight / Price Ratio")
        ax.set_ylabel("Review Score")
        st.pyplot(fig)
    else:
        st.info("Freight or review data missing for freight analysis.")

    st.markdown("---")
    # Payment experience
    st.markdown("### 💳 Payment Experience")
    if "payment_type" in df_filtered.columns and "review_score" in df_filtered.columns:
        pay_choice = st.selectbox("Group by payment type:", options=sorted(df_filtered["payment_type"].dropna().unique()))
        pay_agg = df_filtered[df_filtered["payment_type"] == pay_choice].groupby("payment_type")["review_score"].mean()
        fig, ax = plt.subplots()
        pay_agg.plot(kind="bar", ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("Avg Review Score")
        st.pyplot(fig)
    else:
        st.info("Payment type or review data not available.")

    st.markdown("---")
    # Quick insights and export
    st.subheader("🧠 Quick Insights & Export")
    st.markdown("""
    - Use the segment comparison tab to prioritize audiences for campaigns.  
    - Adjust the late-threshold slider to experiment how late-delivery definitions change KPIs.  
    """)
    with st.expander("📥 Download filtered orders as CSV"):
        csv = df_filtered.to_csv(index=False)
        st.download_button("Download CSV", csv, file_name="filtered_orders.csv", mime="text/csv")


# End of app


