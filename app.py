"""
European Bank - Customer Churn Segmentation Dashboard
Run:  streamlit run app.py     (keep European_Bank.csv in the same folder)
"""
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------
# 1. DEFINITIONS - edit these to change segments (document them in the paper)
# ---------------------------------------------------------------
AGE_BINS, AGE_LABELS = [0, 29, 45, 60, 120], ["<30", "30-45", "46-60", "60+"]
CREDIT_BINS, CREDIT_LABELS = [0, 599, 699, 900], ["Low (<600)", "Medium (600-699)", "High (700+)"]
TENURE_BINS, TENURE_LABELS = [-1, 2, 6, 10], ["New (0-2y)", "Mid-term (3-6y)", "Long-term (7-10y)"]
HIGH_VALUE_QUANTILE = 0.75  # high balance / high-value = top 25% of balances


# ---------------------------------------------------------------
# 2. DATA PREPARATION
# ---------------------------------------------------------------
@st.cache_data
def load_data(path="European_Bank.csv"):
    return prepare(pd.read_csv(path))


def prepare(df):
    df = df.drop(columns=["Surname"], errors="ignore")
    df["AgeGroup"] = pd.cut(df["Age"], AGE_BINS, labels=AGE_LABELS)
    df["CreditBand"] = pd.cut(df["CreditScore"], CREDIT_BINS, labels=CREDIT_LABELS)
    df["TenureGroup"] = pd.cut(df["Tenure"], TENURE_BINS, labels=TENURE_LABELS)
    threshold = df["Balance"].quantile(HIGH_VALUE_QUANTILE)
    df["BalanceSegment"] = "Low-balance"
    df.loc[df["Balance"] == 0, "BalanceSegment"] = "Zero-balance"
    df.loc[df["Balance"] >= threshold, "BalanceSegment"] = "High-balance"
    df["Activity"] = df["IsActiveMember"].map({1: "Active", 0: "Inactive"})
    return df


def churn_table(df, col):
    """Churn rate, size and share of all churners for each segment in `col`."""
    g = df.groupby(col, observed=True)["Exited"].agg(Customers="count", Churned="sum").reset_index()
    g["ChurnRate %"] = (g["Churned"] / g["Customers"] * 100).round(1)
    g["Share of Churners %"] = (g["Churned"] / max(g["Churned"].sum(), 1) * 100).round(1)
    return g


def bar(df, col, title):
    t = churn_table(df, col)
    fig = px.bar(t, x=col, y="ChurnRate %", text="ChurnRate %", title=title, hover_data=["Customers", "Churned"])
    fig.add_hline(y=df["Exited"].mean() * 100, line_dash="dash", annotation_text="Overall")
    return fig


# ---------------------------------------------------------------
# 3. APP
# ---------------------------------------------------------------
st.set_page_config(page_title="Bank Churn Analytics", layout="wide")
st.title("European Bank - Customer Churn Segmentation")

full = load_data()
overall_rate = full["Exited"].mean() * 100

# Sidebar segment filters
st.sidebar.header("Segment filters")
f = full
for label, col in [("Geography", "Geography"), ("Gender", "Gender"), ("Age group", "AgeGroup"),
                   ("Credit band", "CreditBand"), ("Tenure group", "TenureGroup"),
                   ("Balance segment", "BalanceSegment"), ("Activity", "Activity")]:
    options = [x for x in full[col].cat.categories] if str(full[col].dtype) == "category" else sorted(full[col].unique())
    chosen = st.sidebar.multiselect(label, options, default=options)
    f = f[f[col].isin(chosen)]

if f.empty:
    st.warning("No customers match the selected filters.")
    st.stop()

# KPIs (update with the filters)
churn_rate = f["Exited"].mean() * 100
hv = f[f["BalanceSegment"] == "High-balance"]
hv_rate = hv["Exited"].mean() * 100 if len(hv) else float("nan")
act = f[f["Activity"] == "Active"]["Exited"].mean() * 100
inact = f[f["Activity"] == "Inactive"]["Exited"].mean() * 100
engagement_gap = inact - act  # percentage points
balance_at_risk = f.loc[f["Exited"] == 1, "Balance"].sum()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Customers", f"{len(f):,}")
c2.metric("Overall Churn Rate", f"{churn_rate:.1f}%", f"{churn_rate - overall_rate:+.1f} pp vs all")
c3.metric("High-Value Churn Ratio", f"{hv_rate:.1f}%")
c4.metric("Engagement Drop (pp)", f"{engagement_gap:.1f}")
c5.metric("Balance at Risk", f"EUR {balance_at_risk / 1e6:.1f}M")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Geography", "Age & Tenure", "High-Value Explorer"])

with tab1:
    a, b = st.columns(2)
    a.plotly_chart(bar(f, "Gender", "Churn rate by gender"), use_container_width=True)
    b.plotly_chart(bar(f, "Activity", "Churn rate by activity"), use_container_width=True)
    a.plotly_chart(bar(f, "CreditBand", "Churn rate by credit band"), use_container_width=True)
    b.plotly_chart(bar(f, "BalanceSegment", "Churn rate by balance segment"), use_container_width=True)
    st.subheader("Churned vs retained profile (averages)")
    st.dataframe(f.groupby("Exited")[["Age", "CreditScore", "Tenure", "Balance", "EstimatedSalary", "NumOfProducts"]]
                 .mean().round(1).rename(index={0: "Retained", 1: "Churned"}))

with tab2:
    t = churn_table(f, "Geography")
    t["Geographic Risk Index"] = (t["ChurnRate %"] / churn_rate).round(2)  # >1 = riskier than average
    st.plotly_chart(bar(f, "Geography", "Churn rate by country"), use_container_width=True)
    st.dataframe(t, hide_index=True)
    st.subheader("Geography x age interaction")
    heat = f.pivot_table(index="Geography", columns="AgeGroup", values="Exited", aggfunc="mean", observed=True) * 100
    st.plotly_chart(px.imshow(heat.round(1), text_auto=True, color_continuous_scale="Reds",
                              title="Churn rate % (Geography x Age group)"), use_container_width=True)

with tab3:
    a, b = st.columns(2)
    a.plotly_chart(bar(f, "AgeGroup", "Churn rate by age group"), use_container_width=True)
    b.plotly_chart(bar(f, "TenureGroup", "Churn rate by tenure group"), use_container_width=True)
    st.plotly_chart(bar(f, "NumOfProducts", "Churn rate by number of products"), use_container_width=True)

with tab4:
    st.write(f"High-value = top {int((1 - HIGH_VALUE_QUANTILE) * 100)}% of balances "
             f"(>= EUR {full['Balance'].quantile(HIGH_VALUE_QUANTILE):,.0f}).")
    sample = f.sample(min(len(f), 3000), random_state=1).copy()
sample["Status"] = sample["Exited"].map({0: "Retained", 1: "Churned"})
fig = px.scatter(sample, x="EstimatedSalary", y="Balance", color="Status", opacity=0.5,
                 title="Salary vs balance (sample of up to 3,000)")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Drill-down")
    dim = st.selectbox("Break high-balance customers down by", ["Geography", "AgeGroup", "TenureGroup", "Activity", "Gender"])
    st.dataframe(churn_table(hv, dim), hide_index=True)
    st.subheader("Top high-balance churners")
    st.dataframe(hv[hv["Exited"] == 1].sort_values("Balance", ascending=False)
                 [["CustomerId", "Geography", "Age", "Balance", "EstimatedSalary", "NumOfProducts", "Activity"]].head(20),
                 hide_index=True)
