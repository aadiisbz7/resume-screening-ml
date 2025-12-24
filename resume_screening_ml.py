from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("========== AI RESUME SCREENING SYSTEM (ML) ==========")

job_description = input("Enter Job Description: ")

num_resumes = int(input("How many resumes do you want to screen? "))

resumes = []

for i in range(num_resumes):
    print("\nEnter Resume", i + 1)
    resume_text = input()
    resumes.append(resume_text)

documents = [job_description] + resumes

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

jd_vector = tfidf_matrix[0]
resume_vectors = tfidf_matrix[1:]

scores = cosine_similarity(jd_vector, resume_vectors)[0]

results = []

for i in range(num_resumes):
    results.append([i + 1, scores[i]])

results.sort(key=lambda x: x[1], reverse=True)

print("\n========== FINAL SCREENING RESULT ==========")

rank = 1
for r in results:
    print("Rank:", rank)
    print("Resume Number:", r[0])
    print("Match Score:", round(r[1] * 100, 2), "%")
    print("---------------------------")
    rank += 1
