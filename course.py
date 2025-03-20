import pandas as pd
import neattext.functions as nfx
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load Dataset
df = pd.read_csv('filtered_programming_courses.csv')

# Basic EDA
print(df.head())
print(df.info())
print(df.isnull().sum())

# Preprocessing
def preprocess_text(text):
    text = nfx.remove_stopwords(text)
    text = nfx.remove_special_characters(text)
    text = text.lower()
    return text

df['Clean_title'] = df['course_title'].astype(str).apply(preprocess_text)

# Optional - Remove Duplicates
df.drop_duplicates(subset=['Clean_title'], inplace=True)

# Train-Test Split (for testing recommendation accuracy)
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Vectorization using TF-IDF for better results
tfidf = TfidfVectorizer()
tfidf_train = tfidf.fit_transform(train_data['Clean_title'])

# Cosine Similarity Matrix on train set
cosine_sim_train = cosine_similarity(tfidf_train)

# Recommendation Function for Evaluation
def recommend(course_title, k=5):
    course_title = preprocess_text(course_title)
    vect_input = tfidf.transform([course_title])
    sim_scores = cosine_similarity(vect_input, tfidf_train).flatten()
    top_k_idx = sim_scores.argsort()[-k:][::-1]
    recommendations = train_data.iloc[top_k_idx][['course_title']].copy()
    recommendations['score'] = sim_scores[top_k_idx]
    return recommendations

# Example recommendation from train set
print(recommend('Python').head(5))

# Optional: Evaluation - Recommend for test samples and check relevance
test_samples = test_data.sample(5)
for idx, row in test_samples.iterrows():
    print(f"\nInput: {row['course_title']}")
    print(recommend(row['course_title'], k=3))

# Save model components for Flask app
with open('recommendation_model.pkl', 'wb') as f:
    pickle.dump((df, tfidf, cosine_sim_train), f)

print("✅ Model and data saved successfully.")
