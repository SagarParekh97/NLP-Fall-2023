import pandas as pd



loc = './dataset/'
test = pd.read_json(loc + 'test.jsonl', lines=True)
test_label = pd.read_csv(loc + 'sample_prediction.csv')

ids_test = test['id'].tolist()
ids_test_label = test_label['id'].tolist()

print(len(ids_test))
count = 0
for i in ids_test:
    if i in ids_test_label:
        count += 1

print(count)