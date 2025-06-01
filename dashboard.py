import streamlit as st
import pandas as pd
import plotly.express as px
from itertools import chain
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load the clustered data
df = pd.read_csv("clustered_dataset.csv")

st.set_page_config(page_title="Profile Clustering Dashboard", layout="wide")
st.title("Profile Clustering Dashboard <3")
st.markdown("Explore patterns across different profile clusters:)")

# -------------------------------- Combine Skills -------------------------------- #
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

# -------------------------------- PCA Computation -------------------------------- #
numeric_cols = ['monthly income', 'ns2']
categorical_cols = ['ns1', 'ns3', 'gender', 'marital status', 'current employment status',
                    'dbms', 'visualization tools', 'continent', 'occupation_category']

cat_pipeline = OneHotEncoder(handle_unknown='ignore')
num_pipeline = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, numeric_cols),
        ('cat', cat_pipeline, categorical_cols)
    ])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('kmeans', KMeans(n_clusters=4, random_state=42))
])

pipeline.fit(df)
df['cluster'] = pipeline.named_steps['kmeans'].labels_

X_processed = preprocessor.transform(df)
X_dense = X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_dense)

df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

# -------------------------------- Sidebar Filters -------------------------------- #
st.sidebar.header("Filter Data")

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

# -------------------------------- Apply Filters -------------------------------- #
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

# -------------------------------- PCA Scatter Plot -------------------------------- #
st.subheader("🧬 PCA Cluster Visualization (2D)")
fig_pca = px.scatter(
    filtered_df, x='PCA1', y='PCA2', color='cluster',
    title='2D PCA Projection of Clusters',
    hover_data=['monthly income', 'gender', 'continent'],
    color_continuous_scale='Viridis'
)
st.plotly_chart(fig_pca, use_container_width=True)

# -------------------------------- KPIs -------------------------------- #
st.subheader("📊 Summary Statistics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Applicants", len(filtered_df))
col2.metric("Avg. Monthly Income", f"${filtered_df['monthly income'].mean():,.0f}")
col3.metric("Avg. Experience (Years)", f"{filtered_df['experience(in years)'].mean():.1f}")


st.subheader("🔍 Cluster Insights")

# Grouped summaries
for cluster in sorted(filtered_df['cluster'].unique()):
    st.markdown(f"### Cluster {cluster}")

    cluster_df = filtered_df[filtered_df["cluster"] == cluster]

    # Employment distribution
    employment_dist = (
        cluster_df["current employment status"]
        .value_counts(normalize=True)
        .mul(100)
        .round(1)
        .astype(str) + '%'
    )
    
    # Top 3 skills
    top_skills = pd.Series([skill for sublist in cluster_df["skills"] for skill in sublist])
    top_skills = top_skills.value_counts().head(5)

    # Gender & continent split
    gender_split = cluster_df["gender"].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
    continent_split = cluster_df["continent"].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'

    # Top occupation categories
    top_occupations = cluster_df["occupation_category"].value_counts().head(3)

    # Display
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Employment Status**")
        st.write(employment_dist)

    with col2:
        st.markdown("**Top Skills**")
        st.write(top_skills)

    with col3:
        st.markdown("**Gender Split**")
        st.write(gender_split)

    st.markdown("**Top Continents**")
    st.write(continent_split)

    st.markdown("**Most Common Occupation Categories**")
    st.write(top_occupations)

    st.markdown("---")


# -------------------------------- Charts -------------------------------- #
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

fig_line = px.line(
    avg_income_by_exp,
    x="experience(in years)",
    y="monthly income",
    color="cluster",
    markers=True,
    title="📈 Average Monthly Income by Experience and Cluster"
)
fig_line.update_layout(xaxis=dict(dtick=1))
st.plotly_chart(fig_line, use_container_width=True)

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

# -------------------------------- Data Table -------------------------------- #
st.subheader("📄 Filtered Data")
st.dataframe(filtered_df.reset_index(drop=True))
