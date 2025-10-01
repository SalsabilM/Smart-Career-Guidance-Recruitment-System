import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from collections import Counter
from wordcloud import WordCloud


best_model = joblib.load("best_model.pkl")
mlb = joblib.load("mlb.pkl")
le = joblib.load("le.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Career Path Recommendation System")

# Tabs
tab1, tab2 = st.tabs(["📄 Career Recommendation", "📊 Data Visualizations"])

with tab1:
    uploaded_file = st.file_uploader("📄 Upload your CV (TXT file)", type=["txt"])

    if uploaded_file is not None:
        cv_text = uploaded_file.read().decode("utf-8").lower()
        skills_list = [skill.strip() for skill in cv_text.split()]

        st.write("✅ CV uploaded successfully!")

        if st.button("Recommend Career Path"):
            if skills_list:
                skills_vector = mlb.transform([skills_list])
                skills_scaled = scaler.transform(skills_vector)

                probs = best_model.predict_proba(skills_scaled)[0]
                top5_idx = np.argsort(probs)[-5:][::-1]
                top5_specializations = le.inverse_transform(top5_idx)
                top5_probs = probs[top5_idx]

                st.subheader("🔎 Top 5 recommended specializations for this CV:")
                for spec, prob in zip(top5_specializations, top5_probs):
                    st.write(f"- {spec} ({prob:.2%})")
            else:
                st.warning("⚠ No skills detected from the CV.")

with tab2:
    st.subheader("📊 Dataset Visualizations")

    df = pd.read_csv("cleaned.csv")

    # 1. Distribution of match_score
    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(x="match_score", data=df, palette="pastel", ax=ax)
    ax.set_title("Distribution of Match Score")
    st.pyplot(fig)

    # 2. Specialization frequency
    fig, ax = plt.subplots(figsize=(8,4))
    sns.countplot(x="specialization", data=df, palette="pastel", ax=ax)
    ax.set_title("Specialization Counts")
    plt.xticks(rotation=30)
    st.pyplot(fig)

    # 3. Top 10 experience skills
    df['experience_needed_skills_mapped'] = df['experience_needed_skills_mapped'].apply(ast.literal_eval)
    def flatten_skill_lists(column):
        return [skill for sublist in df[column] for skill in sublist]

    experience_skills = flatten_skill_lists("experience_needed_skills_mapped")
    exp_counter = Counter(experience_skills).most_common(10)

    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(x=[c[0] for c in exp_counter], y=[c[1] for c in exp_counter], palette="pastel", ax=ax)
    ax.set_title("Top 10 Experience Skills")
    plt.xticks(rotation=60)
    st.pyplot(fig)

    # Word Cloud for experience_needed_skills_mapped
    st.subheader("Word Cloud of Experience Skills")
    exp_wordcloud = WordCloud(
    width=800, 
    height=400, 
    background_color="white", 
    colormap="Pastel1").generate(" ".join(experience_skills))

    fig, ax = plt.subplots(figsize=(10,6))
    ax.imshow(exp_wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    # 4. Top 10 resume skills
    df['resume_skills_mapped'] = df['resume_skills_mapped'].apply(ast.literal_eval)
    resume_skills = flatten_skill_lists("resume_skills_mapped")
    resume_counter = Counter(resume_skills).most_common(10)

    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(x=[c[0] for c in resume_counter], y=[c[1] for c in resume_counter], palette="pastel", ax=ax)
    ax.set_title("Top 10 Resume Skills")
    plt.xticks(rotation=60)
    st.pyplot(fig)

    # 5. Skill overlap vs match_score
    df['skill_overlap'] = df.apply(lambda x: len(set(x['resume_skills_mapped']) & set(x['experience_needed_skills_mapped'])), axis=1)

    fig, ax = plt.subplots()
    sns.scatterplot(x='skill_overlap', y='match_score', data=df, ax=ax)
    ax.set_title('Match Score vs Skill Overlap')
    st.pyplot(fig)