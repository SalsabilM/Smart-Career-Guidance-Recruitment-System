import pandas as pd
import string
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize

# ------------------- Skill Mapping Dictionary -------------------
skill_mapping = {
    # ------------------- Data Scientist -------------------
    # Programming
    "python": "Python",
    "r": "R",
    "sql": "Structured Query Language",
    "sas": "SAS",
    "scala": "Scala",
    "java": "Java",

    # Math & Stats
    "linear algebra": "Linear Algebra",
    "calculus": "Calculus",
    "probability": "Probability & Statistics",
    "statistics": "Probability & Statistics",
    "hypothesis testing": "Hypothesis Testing",
    "bayesian": "Bayesian Methods",
    "experimental design": "Experimental Design",

    # ML
    "supervised learning": "Supervised Learning",
    "unsupervised learning": "Unsupervised Learning",
    "reinforcement learning": "Reinforcement Learning",
    "deep learning": "Deep Learning",
    "nlp": "Natural Language Processing",
    "natural language processing": "Natural Language Processing",
    "computer vision": "Computer Vision",
    "recommendation systems": "Recommendation Systems",
    "feature engineering": "Feature Engineering",
    "model evaluation": "Model Evaluation & Validation",
    "validation": "Model Evaluation & Validation",

    # Data Management
    "data cleaning": "Data Cleaning",
    "data wrangling": "Data Cleaning",
    "data transformation": "Data Transformation",
    "data integration": "Data Integration",
    "data warehouse": "Data Warehousing",
    "big data": "Big Data (Hadoop, Spark)",
    "hadoop": "Hadoop",
    "spark": "Apache Spark",
    "etl": "ETL Pipelines",

    # Visualization
    "matplotlib": "Matplotlib",
    "seaborn": "Seaborn",
    "plotly": "Plotly",
    "powerbi": "Microsoft Power BI",
    "power bi": "Microsoft Power BI",
    "tableau": "Tableau",
    "excel": "Microsoft Excel",
    "looker": "Looker",

    # Frameworks
    "tensorflow": "TensorFlow",
    "pytorch": "PyTorch",
    "keras": "Keras",
    "scikit-learn": "Scikit-learn",
    "sklearn": "Scikit-learn",
    "nltk": "NLTK",
    "spacy": "SpaCy",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",

    # Data Engineering
    "airflow": "Apache Airflow",
    "docker": "Docker",
    "kubernetes": "Kubernetes",

    # Cloud
    "aws": "AWS Cloud",
    "sagemaker": "AWS SageMaker",
    "redshift": "AWS Redshift",
    "gcp": "Google Cloud Platform",
    "bigquery": "Google BigQuery",
    "azure": "Microsoft Azure",
    "synapse": "Azure Synapse",

    # Soft Skills
    "critical thinking": "Critical Thinking",
    "problem solving": "Problem Solving",
    "communication": "Communication",
    "collaboration": "Collaboration",
    "business acumen": "Business Acumen",
    "storytelling": "Storytelling with Data",

    # Others
    "ab testing": "A/B Testing",
    "a/b testing": "A/B Testing",
    "time series": "Time Series Analysis",
    "mlops": "Machine Learning Operations",
    "git": "Git",
    "github": "GitHub",
    "gitlab": "GitLab",
    "version control": "Version Control",

    # ------------------- Data Analyst -------------------
    "qlik": "QlikView",
    "regression": "Regression Analysis",
    "mysql": "MySQL",
    "postgresql": "PostgreSQL",
    "oracle": "Oracle Database",
    "snowflake": "Snowflake",

    # ------------------- Software Engineer -------------------
    "c++": "C++",
    "c#": "C#",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "ruby": "Ruby",
    "go": "GoLang",
    "php": "PHP",
    "spring": "Spring Framework",
    "spring boot": "Spring Boot",
    "django": "Django",
    "flask": "Flask",
    "react": "React.js",
    "angular": "Angular",
    "vue": "Vue.js",
    "nodejs": "Node.js",
    "express": "Express.js",
    "jenkins": "Jenkins",
    "ci/cd": "CI/CD Pipelines",
    "mongodb": "MongoDB",
    "redis": "Redis",
    "oop": "Object-Oriented Programming",
    "design patterns": "Software Design Patterns",
    "system design": "System Design",
    "rest": "REST APIs",
    "graphql": "GraphQL",
    "teamwork": "Collaboration",

    # ------------------- ML Engineer -------------------
    "ml": "Machine Learning",
    "machine learning": "Machine Learning",
    "cv": "Computer Vision",
    "model deployment": "Model Deployment",
    "onnx": "ONNX",
    "vertex ai": "Google Vertex AI",
    "azure ml": "Azure Machine Learning",

    # ------------------- Product Manager -------------------
    "jira": "Jira",
    "confluence": "Confluence",
    "trello": "Trello",
    "slack": "Slack",
    "powerpoint": "Microsoft PowerPoint",
    "roadmap": "Product Roadmapping",
    "backlog": "Product Backlog Management",
    "mvp": "Minimum Viable Product",
    "okrs": "Objectives and Key Results",
    "kpis": "Key Performance Indicators",
    "market research": "Market Research",
    "user research": "User Research",
    "ux": "User Experience",
    "ui": "User Interface Design",
    "wireframing": "Wireframing",
    "prototyping": "Prototyping",
    "scrum": "Scrum Methodology",
    "kanban": "Kanban Methodology",
    "agile": "Agile Practices",
    "leadership": "Leadership",
    "stakeholder management": "Stakeholder Management",
    "decision making": "Decision Making",
    "negotiation": "Negotiation",
    "strategic thinking": "Strategic Thinking"
}


def map_skills(skills):
    normalized = []
    for s in skills:
        key = s.strip().lower()
        if key in skill_mapping:
            normalized.append(skill_mapping[key])
        else:
            normalized.append(s)
    return normalized


if __name__ == "__main__":
    # -------------------------------
    # Load dataset
    # -------------------------------
    df = pd.read_csv("resume_job_matching_dataset.csv")
    print(df.head())
    # -------------------------------
    # Step 1: Extract specialization and experience
    # -------------------------------
    def extract_parts(text):
        specialization, experience_needed = None, None
        if isinstance(text, str):
            spec_match = re.search(r'^(.*?)\s+needed', text, re.IGNORECASE)
            if spec_match:
                specialization = spec_match.group(1).strip()

            experience_match = re.search(r'needed with experience in\s+(.*)', text, re.IGNORECASE)
            if experience_match:
                experience_needed = experience_match.group(1).strip()

        return pd.Series([specialization, experience_needed])

    df[['specialization', 'experience_needed']] = df['job_description'].apply(extract_parts)
    df = df.drop(columns=['job_description'])

    # -------------------------------
    # Step 2: NLTK setup
    # -------------------------------
    # nltk.download('wordnet')
    # nltk.download('punkt')
    # nltk.download('stopwords')
    # nltk.download('averaged_perceptron_tagger')

    lemmatizer = WordNetLemmatizer()
    stop_words = set(nltk.corpus.stopwords.words('english'))

    # Extra filler words to remove
    filler_words = {
        "experienced", "professional", "skilled", "ability", "knowledge", "proficient",
        "working", "familiar", "strong", "good", "excellent", "understanding",
        "etc", "including", "background", "competency", "capability"
    }

    # -------------------------------
    # Step 3: Define multi-word skills
    # -------------------------------
    multi_word_skills = [
        "power bi", "machine learning", "deep learning", "natural language processing",
        "data science", "computer vision", "time series", "project management",
        "cloud computing", "big data", "data analysis", "artificial intelligence"
    ]

    # -------------------------------
    # Step 4: Helper functions
    # -------------------------------
    def get_wordnet_pos(word):
        """Map POS tags to WordNet POS."""
        tag = pos_tag([word])[0][1][0].upper()
        tag_dict = {
            'J': wordnet.ADJ,
            'N': wordnet.NOUN,
            'V': wordnet.VERB,
            'R': wordnet.ADV
        }
        return tag_dict.get(tag, wordnet.NOUN)

    def smart_lemmatize(word):
        """Lemmatize effectively for verbs/nouns/adjectives."""
        lemma = lemmatizer.lemmatize(word, get_wordnet_pos(word))
        if lemma == word and word.endswith("ing"):
            lemma = lemmatizer.lemmatize(word, wordnet.VERB)
        return lemma

    def clean_and_split(text):
        """Clean text and return list of skill keywords (preserve multi-word skills)."""
        if isinstance(text, str):
            text = text.lower()

            # Replace multi-word skills with underscore versions
            for phrase in multi_word_skills:
                text = text.replace(phrase, phrase.replace(" ", "_"))

            text = re.sub(r'\d+', '', text)  # remove numbers
            text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
            words = word_tokenize(text)

            # Restore multi-word phrases
            words = [w.replace("_", " ") for w in words]

            # Remove stopwords + filler words
            words = [w for w in words if w not in stop_words and w not in filler_words]

            # Lemmatize single-word terms only
            final_words = []
            for w in words:
                if " " in w:
                    final_words.append(w)
                else:
                    final_words.append(smart_lemmatize(w))

            return list(dict.fromkeys(final_words))  # remove duplicates, preserve order
        return []

    # -------------------------------
    # Step 5: Apply text cleaning
    # -------------------------------
    df['resume_skills'] = df['resume'].apply(clean_and_split)
    df['experience_needed_skills'] = df['experience_needed'].apply(clean_and_split)

    df = df.drop(columns=['resume', 'experience_needed'])

    # -------------------------------
    # Step 6: Filter out noise words
    # -------------------------------
    noise_words = {"well", "try", "return", "indicate", "char", "wait", "democrat"}

    df["resume_skills_cleaned"] = df["resume_skills"].apply(
        lambda skills: [s for s in skills if s.lower() not in noise_words] if isinstance(skills, list) else skills
    )

    print(df[["resume_skills", "resume_skills_cleaned"]].head())

    df = df.drop(columns=['resume_skills'])

    # -------------------------------
    # Step 7: Map skills
    # -------------------------------
    df["resume_skills_mapped"] = df.apply(lambda row: map_skills(row["resume_skills_cleaned"]), axis=1)
    df["experience_needed_skills_mapped"] = df.apply(lambda row: map_skills(row["experience_needed_skills"]), axis=1)

    df = df.drop(columns=['resume_skills_cleaned', 'experience_needed_skills'])

    # -------------------------------
    # Step 8: Handle duplicates
    # -------------------------------
    df["resume_skills_mapped"] = df["resume_skills_mapped"].apply(tuple)
    df["experience_needed_skills_mapped"] = df["experience_needed_skills_mapped"].apply(tuple)

    print("Number of duplicates:", df.duplicated().sum())
    # Uncomment to remove duplicates
    # df = df.drop_duplicates()

    # -------------------------------
    # Save cleaned dataset
    # -------------------------------
    df.to_csv('cleaned.csv', index=False)
