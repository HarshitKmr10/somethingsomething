# somethingsomething

import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import joblib
import lightgbm as lgb
from sklearn.metrics import accuracy_score as acc

def preprocess_text(text):
    text = re.sub(r"https?://\S+|www.\S+", "", text)
    text = re.sub(r'[^\x00-\x7f]', r'', text)
    html = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    return re.sub(html, "", text)

def load_test_data(file_path):
    df = pd.read_xlsv(file_path)
    df['Meeting Notes'] = df['Meeting Notes'].astype(str)
    df = df[['Meeting Notes', "Topic"]]
    return df

def apply_model_and_threshold(model, df, threshold):
    x = df["Meeting Notes"].apply(preprocess_text)
    probabilities = model.predict_proba(x)
    le = LE.fit(df["Topic"])
    topic_names = le.inverse_transform(np.arange(len(model.classes_)))
    predicted_classes = [topic_names[i] for i in np.argmax(probabilities, axis=1)]
    filtered_classes = [cls if prob > threshold else 'NAN' for cls, prob in zip(predicted_classes, np.max(probabilities, axis=1))]
    result_df = pd.DataFrame(probabilities, columns=topic_names)
    df["predictedtopic"] = filtered_classes
    df2 = df[df["predictedtopic"] != "NAN"]
    num_records_df2 = len(df2)
    print(type(df2["predictedtopic"]))
    print(type(df2["Topic"]))
    train_acc = acc(le.transform(df2["predictedtopic"]), le.transform(df2['Topic']))
    df = pd.concat([df, result_df], axis=1)
    return df, num_records_df2, train_acc

def save_final_results(df, model, threshold):
    probabilities = model.predict_proba(df["Meeting Notes"].apply(preprocess_text))
    filtered_prob = [prob if prob > threshold else "NAN" for prob in np.max(probabilities, axis=1)]
    df["probability"] = filtered_prob
    df.to_excel('final_results.xlsx', index=False)

# Initialize LabelEncoder
LE = LabelEncoder()

# Load test data
test_data = load_test_data("path")

# Load pre-trained model
loaded_model = joblib.load("model.pkl")

# Set threshold for prediction probabilities
prediction_threshold = 0.75

# Apply model and threshold, calculate results
processed_data, num_records, accuracy = apply_model_and_threshold(loaded_model, test_data, prediction_threshold)
print("Number of records in df2:", num_records)
print("Accuracy:", accuracy)

# Save final results to Excel file
save_final_results(processed_data, loaded_model, prediction_threshold)
