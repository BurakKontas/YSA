import pandas as pd
import numpy as np

df = pd.read_csv("r_squared.csv")

# times = df['time_taken']

# total_time = times.sum()

# learning_rate kolonunu .10f formatına çevirme
df['learning_rate'] = df['learning_rate'].apply(lambda x: format(x, '.10f'))


grouped = df.groupby("learning_rate")

means = grouped.mean()

# turn every floating data to .15f
# Tüm float değerleri .15f formatına çevirme
means = means.applymap(lambda x: format(x, '.10f') if not isinstance(x, str) else x)

# save means as new csv
means.to_csv("means.csv")