import pandas as pd
from flaml import AutoML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from joblib import dump,load

# 加载数据
data = pd.read_csv("datas/knowledge_base.csv")

# 使用TF-IDF向量化问题文本
vectorizer = TfidfVectorizer()
X_transformed = vectorizer.fit_transform(data['Question'])

# 使用LabelEncoder转换答案为整数标签
label_encoder = LabelEncoder()
y_transformed = label_encoder.fit_transform(data['Answer'])

# 查看数据
print(data.head())

automl = AutoML()

automl_settings = {
    "time_budget": 300,  # 总训练时间为300秒
    "metric": 'accuracy',  # 评估标准为准确度
    "task": 'classification',  # 任务类型为分类
    "log_file_name": "flaml.log",  # 日志文件名
}

# 传入转换后的数据集，FLAML 将自动处理训练和验证
automl.fit(X_train=X_transformed, y_train=y_transformed, **automl_settings)

# 示例问题
sample_question = "大数据指的是什么？"
sample_question_transformed = vectorizer.transform([sample_question])
predicted_answer_idx = automl.predict(sample_question_transformed)
predicted_answer = label_encoder.inverse_transform([predicted_answer_idx])[0]
print("Predicted Answer:", predicted_answer)
dump(automl, 'trained_model.joblib')