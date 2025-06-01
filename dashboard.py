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

# -------------------------------- Original Cluster Plots -------------------------------- #
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

# Convert 'experience(in years)' to numeric just in case
filtered_df["experience(in years)"] = pd.to_numeric(filtered_df["experience(in years)"], errors='coerce')

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

# -------------------------------- Cluster Insights -------------------------------- #
st.subheader("Cluster Insights B-)")

# New visual insights
# Employment rate
employment_rate = filtered_df.groupby('cluster')['current employment status'].apply(lambda x: (x == 'employed').mean()).reset_index(name='employment_rate')
st.plotly_chart(px.bar(employment_rate, x='cluster', y='employment_rate', title='Employment Rate by Cluster'))

# Top gender by cluster
gender_dist = filtered_df.groupby(['cluster', 'gender']).size().reset_index(name='count')
st.plotly_chart(px.bar(gender_dist, x='cluster', y='count', color='gender', title='Gender Distribution by Cluster'))

# Continent
continent_dist = filtered_df.groupby(['cluster', 'continent']).size().reset_index(name='count')
st.plotly_chart(px.bar(continent_dist, x='cluster', y='count', color='continent', title='Continent Split by Cluster'))

# Occupation category
occ_dist = filtered_df.groupby(['cluster', 'occupation_category']).size().reset_index(name='count')
st.plotly_chart(px.bar(occ_dist, x='cluster', y='count', color='occupation_category', title='Top Occupation Categories by Cluster'))


# -------------------------------- Data Table -------------------------------- #
st.subheader("📄 Filtered Data")
st.dataframe(filtered_df.reset_index(drop=True))

