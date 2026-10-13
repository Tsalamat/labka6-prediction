# -*- coding: utf-8 -*-
# Practice 6: salary prediction with TF-IDF and Ridge
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

# зафиксируем random state для воспроизводимости результатов
RANDOM_STATE = 42

DATA_URL = "https://raw.githubusercontent.com/evgpat/edu_stepik_first_step_in_nlp/main/datasets/salary_data_cleaned.csv"
from pathlib import Path
# Можно положить salary_data_cleaned.csv рядом с ноутбуком для запуска без интернета.
df = pd.read_csv("salary_data_cleaned.csv" if Path("salary_data_cleaned.csv").exists() else DATA_URL)

print("Размер датасета:", df.shape)
print("Колонки:", df.columns.tolist())
print(df.head().to_string(index=False))
print("Пропуски:\n", df[["Job Description", "avg_salary"]].isna().sum())

salary = pd.to_numeric(df["avg_salary"], errors="coerce")
plt.figure(figsize=(9, 5))
plt.hist(salary.dropna(), bins=35, edgecolor="black", alpha=0.8)
plt.title("Распределение средней зарплаты")
plt.xlabel("avg_salary")
plt.ylabel("Количество вакансий")
plt.tight_layout()
plt.savefig("salary_distribution.png", dpi=150)
plt.show()

data = df[["Job Description", "avg_salary"]].copy()
data["avg_salary"] = pd.to_numeric(data["avg_salary"], errors="coerce")
data = data.dropna(subset=["Job Description", "avg_salary"])
data = data[data["Job Description"].astype(str).str.strip().ne("")]
trX, teX, trY, teY = train_test_split(
    data["Job Description"].astype(str), data["avg_salary"],
    test_size=0.2, random_state=RANDOM_STATE
)
print("Train:", len(trX), "Test:", len(teX))

vectorizer = TfidfVectorizer(min_df=5, max_df=1.0, ngram_range=(1, 1))
trX_tfidf = vectorizer.fit_transform(trX)
teX_tfidf = vectorizer.transform(teX)
print("Размер TF-IDF train:", trX_tfidf.shape)
print("Ответ — число столбцов:", trX_tfidf.shape[1])

model = Ridge(alpha=1.1)
model.fit(trX_tfidf, trY)
train_preds = model.predict(trX_tfidf)
test_preds = model.predict(teX_tfidf)

train_r2 = r2_score(trY, train_preds)
test_r2 = r2_score(teY, test_preds)
print(f"R² train: {train_r2:.6f}")
print(f"R² test:  {test_r2:.6f}")
plt.figure(figsize=(7, 6))
plt.scatter(teY, test_preds, alpha=0.4)
lo, hi = min(teY.min(), test_preds.min()), max(teY.max(), test_preds.max())
plt.plot([lo, hi], [lo, hi], linestyle="--", color="black")
plt.xlabel("Фактическая зарплата")
plt.ylabel("Предсказанная зарплата")
plt.title("Ridge: предсказанные и фактические зарплаты")
plt.tight_layout()
plt.savefig("prediction_scatter.png", dpi=150)
plt.show()

def predict_salary(job_text):
    return float(model.predict(vectorizer.transform([job_text]))[0])
print("director:", predict_salary("director"))

print("manager:", predict_salary("manager"))

data_scientist_salary = predict_salary("data scientist")
print("data scientist:", data_scientist_salary)

feature_names = vectorizer.get_feature_names_out()
weights = model.coef_.ravel()
max_idx = int(np.argmax(weights))
print("Самый положительный признак:", feature_names[max_idx])
print("Его вес:", weights[max_idx])
print("Топ-10 признаков с положительными весами:")
print(pd.DataFrame({"feature": feature_names, "weight": weights})
      .sort_values("weight", ascending=False).head(10).to_string(index=False))

# Сохраняем все ответы для отчета и проверки.
answers = pd.DataFrame([{
    "tfidf_columns": trX_tfidf.shape[1],
    "train_r2": train_r2, "test_r2": test_r2,
    "director_salary": predict_salary("director"),
    "manager_salary": predict_salary("manager"),
    "data_scientist_salary": data_scientist_salary,
    "strongest_positive_feature": feature_names[max_idx],
    "strongest_positive_weight": weights[max_idx],
}])
answers.to_csv("answers.csv", index=False)
print("Ответы сохранены в answers.csv")
