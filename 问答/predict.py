from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# 加载模型
model = load('trained_model.joblib')
label_encoder = load('label_encoder.joblib')
vectorizer = load('tfidf_vectorizer.joblib')

# 使用加载的模型进行预测
sample_question = "什么是人工智能？"

sample_question_transformed = vectorizer.transform([sample_question])
predicted_answer_idx = model.predict(sample_question_transformed)
predicted_answer = label_encoder.inverse_transform([predicted_answer_idx])[0]
print("Predicted Answer:", predicted_answer)