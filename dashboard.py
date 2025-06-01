import streamlit as st
import pandas as pd
import plotly.express as px
from itertools import chain

# Load the clustered data
df = pd.read_csv("clustered_dataset.csv")

st.set_page_config(page_title="Profile Clustering Dashboard", layout="wide")
st.title("Profile Clustering Dashboard <3")
st.markdown("Explore patterns across different profile clusters:)")

# Sidebar filters with grouping and reordered

st.sidebar.header("Filter Data")

def combine_skills(row):
    dbms = row["dbms"]
    vis = row["visualization tools"]
    
    skills = []
    if pd.notna(dbms) and dbms.lower() != "missing":
        skills.extend([s.strip() for s in dbms.split(",")])
    if pd.notna(vis) and vis.lower() != "missing":
        skills.extend([s.strip() for s in vis.split(",")])
    
    return skills if skills else ["Missing"]

df["skills"] = df.apply(combine_skills, axis=1)
all_skills = sorted(set(chain.from_iterable(df["skills"])))

# Group: Main Clustering
with st.sidebar.expander("Main Clustering", expanded=True):
    cluster_filter = st.multiselect("Select Cluster", sorted(df["cluster"].unique()))

# Group: Occupation & Employment
with st.sidebar.expander("Occupation & Employment", expanded=True):
    occupation_category_filter = st.multiselect("Select Occupation Category", sorted(df["occupation_category"].unique()))
    employment_filter = st.multiselect("Employment Status", df["current employment status"].unique())
    skills_filter = st.multiselect("Select Skills", all_skills)

# Group: Demographics
with st.sidebar.expander("Demographics", expanded=True):
    gender_filter = st.multiselect("Select Gender", df["gender"].unique())
    marital_status_filter = st.multiselect("Marital Status", df["marital status"].unique())

# Group: Location
with st.sidebar.expander("Location", expanded=True):
    continent_filter = st.multiselect("Select Continent", df["continent"].unique())

# Apply filters
filtered_df = df.copy()

if cluster_filter:
    filtered_df = filtered_df[filtered_df["cluster"].isin(cluster_filter)]
if occupation_category_filter:
    filtered_df = filtered_df[filtered_df["occupation_category"].isin(occupation_category_filter)]
if employment_filter:
    filtered_df = filtered_df[filtered_df["current employment status"].isin(employment_filter)]
if skills_filter:
    filtered_df = filtered_df[filtered_df["skills"].apply(lambda skills: any(skill in skills for skill in skills_filter))]
if gender_filter:
    filtered_df = filtered_df[filtered_df["gender"].isin(gender_filter)]
if marital_status_filter:
    filtered_df = filtered_df[filtered_df["marital status"].isin(marital_status_filter)]
if continent_filter:
    filtered_df = filtered_df[filtered_df["continent"].isin(continent_filter)]

# KPIs
st.subheader("📊 Summary Statistics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Applicants", len(filtered_df))
col2.metric("Avg. Monthly Income", f"${filtered_df['monthly income'].mean():,.0f}")
col3.metric("Avg. Experience (Years)", f"{filtered_df['experience(in years)'].mean():.1f}")

# Pie Chart - Cluster distribution
st.plotly_chart(
    px.pie(filtered_df, names='cluster', title='Cluster Distribution'),
    use_container_width=True
)

# Bar Chart - Avg. Income by Cluster
st.plotly_chart(
    px.bar(
        filtered_df.groupby("cluster")["monthly income"].mean().reset_index(),
        x="cluster", y="monthly income",
        title="Average Monthly Income by Cluster", text_auto=True
    ),
    use_container_width=True
)

# Line Chart - Avg. Income by Experience and Cluster
avg_income_by_exp = (
    filtered_df
    .groupby(["experience(in years)", "cluster"])["monthly income"]
    .mean()
    .reset_index()
)

fig = px.line(
    avg_income_by_exp,
    x="experience(in years)",
    y="monthly income",
    color="cluster",
    markers=True,
    title="📈 Average Monthly Income by Experience and Cluster",
)

fig.update_layout(xaxis=dict(dtick=1))  # Show all experience years as ticks
st.plotly_chart(fig, use_container_width=True)

# Histogram - Age Distribution
st.plotly_chart(
    px.histogram(
        filtered_df,
        x="age",
        color="cluster",
        barmode="overlay",
        nbins=30,
        title="Age Distribution by Cluster"
    ),
    use_container_width=True
)

# Final Data Table
st.subheader("📄 Filtered Data")
st.dataframe(filtered_df.reset_index(drop=True))

