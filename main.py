import pandas as pd
import os
from sklearn.model_selection import train_test_split

SPAM = 1
HAM = 0
TEST_SIZE = 0.3

FILE_PATHS = [(os.path.abspath(
    os.path.dirname(__file__)+'/Data_set/spam_2/'), SPAM),
    (os.path.abspath(
        os.path.dirname(__file__)+'/Data_set/easy_ham/'), HAM),
    (os.path.abspath(
        os.path.dirname(__file__)+'/Data_set/hard_ham/'), HAM)
]
SKIP_FILES = ['cmd']

for path, mail_type in FILE_PATHS:
    file_list = os.listdir(path)
    for f in file_list:
        if f in SKIP_FILES:
            continue
        with open(path + f, 'r') as mail:
            content = []
            for line in mail:
                content.append(line.decode('latin-1'))
            content = '\n'.join(content)
        result.append({'data': content, 'type': mail_type})


messages = pd.read_csv(os.path.abspath(
    os.path.dirname(__file__)+'/Data_set/spam.csv'), encoding='latin-1')
print(os.path.abspath(
    os.path.dirname(__file__)+'/Data_set/spam.csv'))
