# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import make_pipeline
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from tqdm.auto import tqdm
import warnings
import joblib  
import os
import fitz 
import requests
from bs4 import BeautifulSoup
import requests



warnings.filterwarnings("ignore")
tqdm.pandas()

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)


# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
data = pd.read_csv(r"C:\Users\91878\Downloads\archive (12)\UpdatedResumeDataSet.csv")
data['processed_resume'] = data['Resume'].progress_apply(preprocess_text)


# Encode categories
label_encoder = LabelEncoder()
data['encoded_category'] = label_encoder.fit_transform(data['Category'])

# Split data
X = data['processed_resume']
y = data['encoded_category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Common TF-IDF parameters
tfidf_params = {
    'tfidfvectorizer__ngram_range': [(1, 1), (1, 2)],
    'tfidfvectorizer__max_df': [0.8, 0.9],
}

# GridSearch 
def run_grid_search(name, pipeline, param_grid):
    print(f"\n Running GridSearchCV for {name}...")
    grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"\n {name} Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_, digits=4))
    return grid.best_estimator_, f1

#  Naive Bayes
nb_pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())
nb_param_grid = {**tfidf_params, 'multinomialnb__alpha': [0.5, 1.0]}
nb_model, f1_nb = run_grid_search("Naive Bayes", nb_pipeline, nb_param_grid)

#  SVM
svm_pipeline = make_pipeline(TfidfVectorizer(), SVC(kernel='linear'))
svm_param_grid = {**tfidf_params, 'svc__C': [0.5, 1.0, 2.0]}
svm_model, f1_svm = run_grid_search("SVM", svm_pipeline, svm_param_grid)

#  Logistic Regression
logreg_pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
logreg_param_grid = {**tfidf_params, 'logisticregression__C': [0.5, 1.0, 2.0]}
logreg_model, f1_logreg = run_grid_search("Logistic Regression", logreg_pipeline, logreg_param_grid)

# Best Model
results = {
    'Naive Bayes': (nb_model, f1_nb),
    'SVM': (svm_model, f1_svm),
    'Logistic Regression': (logreg_model, f1_logreg)
}
best_model_name = max(results, key=lambda k: results[k][1])
best_model, best_f1 = results[best_model_name]

print(f"\n Best Model Based on Macro F1-Score: {best_model_name}")
print(f" F1 Score: {best_f1:.4f}")
print(f" Best Parameters: {best_model.get_params()}")

#  Save
output_dir = "/content/saved_model"
os.makedirs(output_dir, exist_ok=True)

joblib.dump(best_model, os.path.join(output_dir, "best_resume_model.pkl"))
joblib.dump(label_encoder, os.path.join(output_dir, "label_encoder.pkl"))

print(f"\n Model and LabelEncoder saved to: {output_dir}")


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# Load the best model and label encoder
best_model = joblib.load(r"C:\Users\91878\Downloads\best_resume_model.pkl")
label_encoder = joblib.load(r"C:\Users\91878\Downloads\label_encoder.pkl")
pdf_text = extract_text_from_pdf(r"C:\Users\91878\Downloads\Resume_parser_project\curriculum_vitae.pdf")
processed_text = preprocess_text(pdf_text)

# Predict category
predicted_label = best_model.predict([processed_text])[0]
predicted_class = label_encoder.inverse_transform([predicted_label])[0]
print("Predicted Resume Class:", predicted_class)



# Mapping from your predicted class to RemoteOK category
predicted_to_remoteok = {
    "Advocate": "legal",
    "Arts": "design",
    "Automation Testing": "qa",
    "Blockchain": "blockchain",
    "Business Analyst": "analyst",
    "Civil Engineer": "engineering",
    "Data Science": "data-science",
    "Database": "sql",
    "DevOps Engineer": "devops",
    "DotNet Developer": "csharp",
    "ETL Developer": "data-science",
    "Electrical Engineering": "engineering",
    "HR": "hr",
    "Hadoop": "data-science",
    "Health and fitness": "medical",
    "Java Developer": "java",
    "Mechanical Engineer": "engineering",
    "Network Security Engineer": "security",
    "Operations Manager": "ops",
    "PMO": "project-management",
    "Python Developer": "python",
    "SAP Developer": "software-dev",
    "Sales": "sales",
    "Testing": "qa",
    "Web Designing": "design"
}

def scrape_remoteok_jobs(category, keyword=None, num_results=10):
    base_url = f"https://remoteok.com/remote-{category.lower().replace(' ', '-')}-jobs"
    headers = {'User-Agent': 'Mozilla/5.0'}

    response = requests.get(base_url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to fetch jobs for {category}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    job_posts = soup.find_all('tr', class_='job')[:num_results * 2]  # fetch more to filter

    jobs = []
    for post in job_posts:
        title_tag = post.find('h2')
        company_tag = post.find('h3')
        link_tag = post.get('data-href')

        if title_tag and company_tag and link_tag:
            title = title_tag.text.strip()
            job = {
                'title': title,
                'company': company_tag.text.strip(),
                'link': 'https://remoteok.com' + link_tag
            }
            jobs.append(job)

    # Prioritize jobs that contain the keyword in the title
    if keyword:
        keyword_lower = keyword.lower()
        matched = [job for job in jobs if keyword_lower in job['title'].lower()]
        unmatched = [job for job in jobs if keyword_lower not in job['title'].lower()]
        jobs = matched + unmatched

    return jobs[:num_results]


remoteok_category = predicted_to_remoteok.get(predicted_class)
jobs = scrape_remoteok_jobs(remoteok_category, keyword=predicted_class, num_results=10)

# Print results
print(f"\n Top {len(jobs)} jobs for '{predicted_class}' from Remoteok:\n")
for i, job in enumerate(jobs, 1):
    print(f"{i}. {job['title']} at {job['company']}\n   {job['link']}\n")


# Define category mapping to Remotive categories
category_mapping = {
    "Advocate": "legal",
    "Arts": "others",
    "Automation Testing": "qa",
    "Blockchain": "software-dev",
    "Business Analyst": "product",
    "Civil Engineer": "others",
    "Data Science": "data",
    "Database": "software-dev",
    "DevOps Engineer": "devops-sysadmin",
    "DotNet Developer": "software-dev",
    "ETL Developer": "software-dev",
    "Electrical Engineering": "others",
    "HR": "human-resources",
    "Hadoop": "software-dev",
    "Health and fitness": "others",
    "Java Developer": "software-dev",
    "Mechanical Engineer": "others",
    "Network Security Engineer": "devops-sysadmin",
    "Operations Manager": "sales",
    "PMO": "project-management",
    "Python Developer": "software-dev",
    "SAP Developer": "software-dev",
    "Sales": "sales",
    "Testing": "qa",
    "Web Designing": "design"
}

#  Valid Remotive categories for reference (should match API expectations)
remotive_valid_categories = [
    "software-dev", "customer-support", "design", "marketing", "sales",
    "product", "devops-sysadmin", "finance-legal", "hr", "qa", "writing",
    "teaching", "business", "data", "project-management", "others"
]

# Function to fetch jobs from Remotive API for a valid category
def fetch_jobs_remotive(remotive_category, num_results=10):
    category = remotive_category.lower().replace(" ", "-")

    if category not in remotive_valid_categories:
        print(f" '{category}' is not a valid Remotive category.")
        return []

    url = f"https://remotive.com/api/remote-jobs?category={category}"

    try:
        response = requests.get(url)
        response.raise_for_status()

        jobs = response.json().get("jobs", [])
        return [
            {
                "title": job["title"],
                "company": job["company_name"],
                "url": job["url"]
            }
            for job in jobs[:num_results]
        ]
    except Exception as e:
        print(f" Error fetching jobs: {e}")
        return []

# Function to fetch jobs based on predicted class
def fetch_jobs_by_keywords(predicted_class, num_results=10):
    broad_category = category_mapping.get(predicted_class, "others")

    print(f"\n Predicted Class: {predicted_class}")
    print(f" Mapped to Remotive Category: {broad_category}")

    # Try broad category first (Remotive valid category)
    jobs_in_broad_category = fetch_jobs_remotive(broad_category, num_results)
    if jobs_in_broad_category:
        return jobs_in_broad_category

    # Try keyword-based fallback
    keywords = predicted_class.lower().split()
    jobs_with_keywords = []

    for keyword in keywords:
        jobs_with_keywords.extend(fetch_jobs_remotive(keyword, num_results))

    # Remove duplicates by URL
    unique_jobs = {job['url']: job for job in jobs_with_keywords}.values()

    if unique_jobs:
        return list(unique_jobs)

    # Return empty if nothing found
    print(" No jobs found for either mapped category or keywords.")
    return []

jobs = fetch_jobs_by_keywords(predicted_class)

print(f"\n Top {len(jobs)} jobs for '{predicted_class}' from Remotive:\n")
for job in jobs:
    print(f"{job['title']} at {job['company']}\n→ {job['url']}\n")
