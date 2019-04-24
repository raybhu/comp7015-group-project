import pandas as pd
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score
from sklearn.model_selection import cross_val_score

SPAM = 1
HAM = 0
TEST_SIZE = 0.3

FILE_PATHS = [(os.path.abspath(
    os.path.dirname(__file__)+'/spam_and_ham_mails/spam_2/'), SPAM),
    (os.path.abspath(
        os.path.dirname(__file__)+'/spam_and_ham_mails/easy_ham/'), HAM),
    (os.path.abspath(
        os.path.dirname(__file__)+'/spam_and_ham_mails/hard_ham/'), HAM)
]
SKIP_FILES = ['cmd']
result = []
for path, mail_type in FILE_PATHS:
    file_list = os.listdir(path)
    for f in file_list:
        if f in SKIP_FILES:
            continue
        with open(path + '/' + f, 'r', encoding="latin-1") as mail:
            content = []
            for line in mail:
                content.append(line)
            content = '\n'.join(content)
        result.append({'data': content, 'type': mail_type})
data = pd.DataFrame(result)
# X: Data y: Tag
X_train, X_test, y_train, y_test = train_test_split(data['data'], data['type'],
                                                    test_size=TEST_SIZE, random_state=42)
pipeline = Pipeline([
    ('vectorizer',  CountVectorizer()),
    ('classifier',  MultinomialNB())])
pipeline.fit(X_train, y_train)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
print('cross-validation: ', cv_scores)
start = datetime.now()
predicted = pipeline.predict(X_test)  # ['spam', 'ham']
end = datetime.now()
print('time: ', (end - start).total_seconds())  # 1.86
print('confusion matrix: \n', confusion_matrix(y_test, predicted))
print('accuracy_score: ', accuracy_score(y_test, predicted))
print('f1_score: ', f1_score(y_test, predicted))
print('recall_score: ', recall_score(y_test, predicted))
