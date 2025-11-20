# Load csv
import pandas as pd

# Scaling the data 
from sklearn.preprocessing import StandardScaler

# Import KMeans clustering algorithm
from sklearn.cluster import KMeans

# For visualization 
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns


# ----------------------------------------------------
# Step 1: Load dataset and keep only relevant columns
# ----------------------------------------------------
df = pd.read_csv("fifa.csv")

# Keep only the columns that matter
df = df[['Name', 'Overall', 'Potential', 'Value', 'Wage']].dropna()


# ----------------------------------------------------
# Step 2: Convert currency strings into numeric values
# ----------------------------------------------------
def convert_currency(x):
    x = x.replace('€', '')
    if 'M' in x:
        return float(x.replace('M', '')) * 1_000_000
    if 'K' in x:
        return float(x.replace('K', '')) * 1_000
    return float(x)

df['Value'] = df['Value'].apply(convert_currency)
df['Wage'] = df['Wage'].apply(convert_currency)


# ----------------------------------------------------
# Step 3: Create meaningful numeric features
# ----------------------------------------------------
df['AbilityScore'] = (df['Overall'] + df['Potential']) / 2
df['CostScore'] = (df['Value'] + df['Wage']) / 2

# Build the study dataset (only for clustering)
study_df = df[['Name', 'AbilityScore', 'CostScore']].copy()



# ----------------------------------------------------
#                  VISUALIZATIONS
# ----------------------------------------------------
numerical = df[['Overall', 'Potential', 'Value', 'Wage', 'AbilityScore', 'CostScore']]

# --- Histograms ---
numerical.hist(figsize=(12, 8), bins=30)
plt.suptitle("Histograms of All Numeric Features")
plt.show()

# --- Boxplots (Outlier detection) ---
plt.figure(figsize=(12, 6))
sns.boxplot(data=numerical)
plt.title("Boxplot — Outlier Detection")
plt.xticks(rotation=45)
plt.show()

# --- Correlation Heatmap ( ---
plt.figure(figsize=(10, 6))
sns.heatmap(numerical.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# --- Scatter plot Ability vs Cost ---
plt.figure(figsize=(8, 6))
plt.scatter(df['AbilityScore'], df['CostScore'], alpha=0.3)
plt.title("Scatter Plot — Ability vs Cost")
plt.xlabel("AbilityScore")
plt.ylabel("CostScore")
plt.show()






# ----------------------------------------------------
#        Step 4: Scaling before clustering
# ----------------------------------------------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(study_df[['AbilityScore', 'CostScore']])


# -------------------------------------------------------------
#        Step 5: Determine optimal K using Elbow Method
# -------------------------------------------------------------
errors = []
k_range = range(1, 8)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init='auto')
    km.fit(scaled_data)
    errors.append(km.inertia_)

plt.plot(k_range, errors, marker='o')
plt.title("Elbow Method — Ability vs Cost")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia (Error)")
plt.show()


# -------------------------------------------------------------
#   Step 6: Apply KMeans (choose K=4 after elbow analysis)
# -------------------------------------------------------------
kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
study_df.loc[:, 'Cluster'] = kmeans.fit_predict(scaled_data)


# -------------------------------------------------------------------
#    Step 7 (fixed): Assign REAL Matplotlib colors to clusters
# -------------------------------------------------------------------
matplotlib_colors = ['red', 'blue', 'green', 'yellow']

color_map = {
    0: matplotlib_colors[0],
    1: matplotlib_colors[1],
    2: matplotlib_colors[2],
    3: matplotlib_colors[3]
}

study_df.loc[:, "ClusterColor"] = study_df["Cluster"].map(color_map)


# ----------------------------------------------------
#       Step 8 (fixed): Visualize clusters 
# ----------------------------------------------------
plt.figure(figsize=(10, 6))

for cluster in sorted(study_df['Cluster'].unique()):
    cluster_data = study_df[study_df['Cluster'] == cluster]

    plt.scatter(
        cluster_data['AbilityScore'],
        cluster_data['CostScore'],
        s=50,
        color=color_map[cluster],
        label=f"Cluster {cluster} - {color_map[cluster].capitalize()}"
    )

plt.title("Player Clusters Based on Ability vs Cost")
plt.xlabel("AbilityScore")
plt.ylabel("CostScore")
plt.legend()
plt.show()


# ------------------------------------------------------
#          Step 9: Analyze cluster meanings
# ------------------------------------------------------
print("\nCluster Means:")
print(study_df.groupby('Cluster')[['AbilityScore', 'CostScore']].mean())


# --------------------------------------------------------------
#   Step 10: Identify Hidden Gems (High ability + Low cost)
# --------------------------------------------------------------
ability_threshold = study_df['AbilityScore'].quantile(0.60)
cost_threshold = study_df['CostScore'].quantile(0.40)

hidden_gems = study_df[
    (study_df['AbilityScore'] >= ability_threshold) &
    (study_df['CostScore'] <= cost_threshold)
]

hidden_gems = hidden_gems.sort_values("AbilityScore", ascending=False)

print("\nPep Guardiola scouting level:")
print(hidden_gems.head(20))


# ----------------------------------------------------
# FINAL STEP: Save complete dataset with cluster column
# ----------------------------------------------------

# Merge cluster numbers back to the original df
df.loc[:, "Cluster"] = study_df["Cluster"].values

# Export full dataset + cluster number
df.to_csv("fifa_full_with_clusters.csv", index=False)

# Export hidden gems file
hidden_gems.to_csv("hidden_gems_clean.csv", index=False)
