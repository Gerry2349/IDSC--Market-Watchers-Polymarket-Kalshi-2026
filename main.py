import pandas as pd

df = pd.read_csv("data/wallets.csv")
print(len(df.wallet.unique()))
print(len(df.name.unique()))