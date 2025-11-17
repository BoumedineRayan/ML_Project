import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------
# Step 1: Load dataset and keep only relevant columns
# ----------------------------------------------------
df = pd.read_csv("fifa.csv")

# Keep only required columns
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

# Create final clean working dataset with no extra columns
study_df = df[['Name', 'AbilityScore', 'CostScore']]

# ----------------------------------------------------
# Step 4: Scaling before clustering
# ----------------------------------------------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(study_df[['AbilityScore', 'CostScore']])

# ----------------------------------------------------
# Step 5: Determine optimal K using Elbow Method
# ----------------------------------------------------
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

# ----------------------------------------------------
# Step 6: Apply KMeans (choose K=4 after elbow analysis)
# ----------------------------------------------------
kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
study_df['Cluster'] = kmeans.fit_predict(scaled_data)

# ----------------------------------------------------
# Step 7: Visualize clusters
# ----------------------------------------------------
plt.figure(figsize=(10,6))
sns.scatterplot(data=study_df, 
                x='AbilityScore', 
                y='CostScore', 
                hue='Cluster', 
                palette='tab10', 
                s=50)
plt.title("Player Clusters Based on Ability vs Cost")
plt.show()

# ----------------------------------------------------
# Step 8: Analyze cluster meanings
# ----------------------------------------------------
print("\nCluster Means:")
print(study_df.groupby('Cluster')[['AbilityScore', 'CostScore']].mean())

# ----------------------------------------------------
# Step 9: Identify Hidden Gems
# (High ability + Low cost players)
# ----------------------------------------------------
ability_threshold = study_df['AbilityScore'].quantile(0.60)
cost_threshold = study_df['CostScore'].quantile(0.40)

hidden_gems = study_df[(study_df['AbilityScore'] >= ability_threshold) &
                       (study_df['CostScore'] <= cost_threshold)]

hidden_gems = hidden_gems.sort_values("AbilityScore", ascending=False)

print("\n🔥 TOP HIDDEN GEMS:")
print(hidden_gems.head(20))




# Save clean results
hidden_gems.to_csv("hidden_gems_clean.csv", index=False)
print("\nFile exported: hidden_gems_clean.csv")
