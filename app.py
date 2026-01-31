import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Customer Satisfaction Analytics",
    layout="wide"
)

sns.set_style("whitegrid")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("analytics_data.csv")

df = load_data()

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("Filters")

state_filter = st.sidebar.multiselect(
    "Customer State",
    options=sorted(df["customer_state"].unique()),
    default=sorted(df["customer_state"].unique())
)

category_filter = st.sidebar.multiselect(
    "Product Category",
    options=sorted(df["product_category"].unique()),
    default=sorted(df["product_category"].unique())
)

df = df[
    (df["customer_state"].isin(state_filter)) &
    (df["product_category"].isin(category_filter))
]

# -----------------------------
# EXECUTIVE KPIs
# -----------------------------
st.title("📊 Customer Satisfaction Dashboard")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric(
    "Avg Review Score",
    round(df["review_score"].mean(), 2)
)

kpi2.metric(
    "Unhappy Customers %",
    f"{round(df['is_unhappy'].mean() * 100, 2)}%"
)

kpi3.metric(
    "Late Delivery Rate %",
    f"{round((df['delivery_delay_days'] > 0).mean() * 100, 2)}%"
)

kpi4.metric(
    "Avg Freight / Price",
    round(df["freight_to_price_ratio"].mean(), 2)
)

st.markdown("---")

# -----------------------------
# DELIVERY IMPACT
# -----------------------------
st.subheader("🚚 Delivery Performance vs Satisfaction")

col1, col2 = st.columns(2)

with col1:
    delay_review = df.groupby("delay_bucket")["review_score"].mean()
    fig, ax = plt.subplots()
    delay_review.plot(kind="bar", ax=ax)
    ax.set_ylabel("Average Review Score")
    st.pyplot(fig)

with col2:
    delay_unhappy = df.groupby("delay_bucket")["is_unhappy"].mean() * 100
    fig, ax = plt.subplots()
    delay_unhappy.plot(kind="bar", ax=ax, color="red")
    ax.set_ylabel("Unhappy Customers (%)")
    st.pyplot(fig)

st.markdown("---")

# -----------------------------
# COST PERCEPTION
# -----------------------------
st.subheader("💸 Freight Cost Impact")

fig, ax = plt.subplots()
sns.scatterplot(
    data=df,
    x="freight_to_price_ratio",
    y="review_score",
    alpha=0.3,
    ax=ax
)
ax.set_xlabel("Freight / Price Ratio")
ax.set_ylabel("Review Score")
st.pyplot(fig)

st.markdown("---")

# -----------------------------
# PAYMENT EXPERIENCE
# -----------------------------
st.subheader("💳 Payment Method vs Satisfaction")

payment_review = df.groupby("payment_type")["review_score"].mean().sort_values()

fig, ax = plt.subplots()
payment_review.plot(kind="barh", ax=ax)
ax.set_xlabel("Average Review Score")
st.pyplot(fig)

st.markdown("---")

# -----------------------------
# PRODUCT CATEGORY RISK
# -----------------------------
st.subheader("📦 Product Category Satisfaction Risk")

category_stats = (
    df.groupby("product_category")
    .agg(
        avg_review=("review_score", "mean"),
        unhappy_rate=("is_unhappy", "mean"),
        orders=("review_score", "count")
    )
    .query("orders > 200")
    .sort_values("unhappy_rate", ascending=False)
)

st.dataframe(
    category_stats
    .assign(
        unhappy_rate=lambda x: (x["unhappy_rate"] * 100).round(2)
    )
)

st.markdown("---")

# -----------------------------
# GEOGRAPHIC SATISFACTION
# -----------------------------
st.subheader("📍 Satisfaction by State")

state_review = df.groupby("customer_state")["review_score"].mean().sort_values()

fig, ax = plt.subplots(figsize=(10,4))
state_review.plot(kind="bar", ax=ax)
ax.set_ylabel("Average Review Score")
st.pyplot(fig)

# -----------------------------
# INSIGHTS SECTION
# -----------------------------
st.subheader("🧠 Key Business Insights")

st.markdown("""
- Late and very late deliveries show **significantly higher dissatisfaction**
- High freight-to-price ratios reduce perceived value
- Certain product categories contribute disproportionately to bad reviews
- Payment method choice impacts customer satisfaction
- Geographic hotspots exist for low satisfaction
""")
