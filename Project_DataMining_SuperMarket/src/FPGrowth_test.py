from mlxtend.frequent_patterns import fpgrowth
import pandas as pd
import numpy as np
import os
from os.path import dirname, abspath
from mlxtend.preprocessing import TransactionEncoder

BASE_DIR = dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(str(BASE_DIR), "data\store_data.csv")
store_data = pd.read_csv(path, header=None)
records = []
for i in range(0, 7501):
    records.append([str(store_data.values[i, j]) for j in range(0, 20)])

records_withoutNan = []
for i in range(0, len(records)):
    new = []
    for j in range(0, len(records[i])):
        if str(records[i][j]) != "nan":
            new.append(str(records[i][j]))
    records_withoutNan.append(new)

te = TransactionEncoder()
te_ary = te.fit(records_withoutNan).transform(records_withoutNan)
df = pd.DataFrame(te_ary, columns=te.columns_)
# print(df)
result = fpgrowth(df, min_support=0.06, use_colnames=True)
print(result)