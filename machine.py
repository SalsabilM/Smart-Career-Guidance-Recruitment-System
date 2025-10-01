import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from clean import map_skills
import joblib

# ======================
# 1. Load Data
# ======================
df = pd.read_csv("cleaned.csv")
# df = df.drop(columns=['Unnamed: 0'])

print(df.head())

# ======================
# 2. Parse list-like strings
# ======================
def parse_list_col(x):
    """Parse a column that may contain list-like strings into a Python list."""
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x)
    except Exception:
        return [i.strip().strip("'\"") for i in str(x).split(",") if i.strip()]

df["resume_skills_mapped"] = df["resume_skills_mapped"].apply(parse_list_col)
df["experience_needed_skills_mapped"] = df["experience_needed_skills_mapped"].apply(parse_list_col)

# Combine skills
df["skills"] = df.apply(lambda r: list(dict.fromkeys(r["resume_skills_mapped"])), axis=1)

# ======================
# 3. Prepare Features & Target
# ======================
y = df["specialization"]
le = LabelEncoder()
y_enc = le.fit_transform(y)

mlb = MultiLabelBinarizer(sparse_output=False)
X_skills = mlb.fit_transform(df["skills"])

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X_skills)

# Train/test split for evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)

# ======================
# 4. Cross-validation
# ======================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
}

# ======================
# 5. Clustering (unsupervised analysis)
# ======================
# PCA for dimensionality reduction
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

# --- KMeans ---
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["kmeans_cluster"] = kmeans.fit_predict(X)

# --- Agglomerative ---
agg = AgglomerativeClustering(n_clusters=5)
df["agg_cluster"] = agg.fit_predict(X)

# ======================
# 6. Cross-validation & Model Selection
# ======================
best_model_name = None
best_score = -1
best_model = None

for name, model in models.items():
    scores = cross_val_score(model, X, y_enc, cv=skf, scoring="accuracy")
    mean_score = scores.mean()
    print(f"{name} Accuracy: {mean_score:.4f} ± {scores.std():.4f}")
    print("Scores:", scores)
    print("=" * 50)

    if mean_score > best_score:
        best_score = mean_score
        best_model_name = name
        best_model = model

print(f"✅ Best model selected: {best_model_name} with CV accuracy = {best_score:.4f}")

# ======================
# 7. Train best model & evaluate on test set
# ======================
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# Decode labels
y_pred_labels = le.inverse_transform(y_pred)
y_test_labels = le.inverse_transform(y_test)

# Print first 5 predictions for comparison
print("Predicted Specializations:", y_pred_labels[:5])
print("Actual Specializations:", y_test_labels[:5])

# ======================
# Visualization
# ======================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# KMeans plot
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=df["kmeans_cluster"], cmap="tab10", s=20)
axes[0].set_title("KMeans Clusters")

# Agglomerative plot
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=df["agg_cluster"], cmap="tab10", s=20)
axes[1].set_title("Agglomerative Clusters")

plt.show()

# ======================
# 8. Recommendation Function
# ======================
def recommend_top_specializations(new_skills, model, mlb, le, scaler, top_n=5):
    """
    Recommend the top specializations for a new CV based on its skills.

    Parameters:
    - new_skills: list of skills from the new CV
    - model: trained model
    - mlb: fitted MultiLabelBinarizer
    - le: fitted LabelEncoder
    - scaler: fitted StandardScaler
    - top_n: number of recommendations to return
    """
    # Normalize skills using map_skills
    normalized_skills = map_skills(new_skills)

    # Keep only skills that exist in training data
    filtered_skills = [s for s in normalized_skills if s in mlb.classes_]

    if not filtered_skills:
        print("No matching skills found. Consider improving your skill set and try again.")
        return []

    # Transform skills into model input format
    new_X = mlb.transform([filtered_skills])

    # Standardize using the fitted scaler
    new_X = scaler.transform(new_X)

    # Predict probabilities
    probs = model.predict_proba(new_X)[0]

    # Sort classes by probability
    top_indices = np.argsort(probs)[::-1][:top_n]
    top_specializations = le.inverse_transform(top_indices)
    top_probs = probs[top_indices]

    return list(zip(top_specializations, top_probs))


# Example usage (commented out for now):
# user_input = input("Enter your skills separated by commas: ")
# new_cv_skills = [s.strip() for s in user_input.split(",") if s.strip()]
# recommendations = recommend_top_specializations(new_cv_skills, best_model, mlb, le, scaler, top_n=5)
# for spec, prob in recommendations:
#     print(f"- {spec} (probability: {prob:.2f})")

# ======================
# 9. Save Model & Encoders
# ======================
joblib.dump(best_model, "best_model.pkl")
joblib.dump(mlb, "mlb.pkl")
joblib.dump(le, "le.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Saved: best_model.pkl, mlb.pkl, le.pkl, scaler.pkl")
