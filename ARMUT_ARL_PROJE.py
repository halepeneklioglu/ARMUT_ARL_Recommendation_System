import pandas as pd
from tomlkit import datetime

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules
import datetime

df_ = pd.read_csv("armut_data.csv")
df = df_.copy()
df.info()
df.describe().T

df["UserId"] = df["UserId"].astype("object")

df.head()
df["Hizmet"] = (df["ServiceId"]).astype(str) + "_" + (df["CategoryId"]).astype(str)

df["CreateDate"] = pd.to_datetime(df["CreateDate"])

df["New_Date"] = df["CreateDate"].dt.strftime('%Y-%m')

df["SepetID"] = (df["UserId"]).astype(str) + "_" + (df["New_Date"]).astype(str)

df_pivot = df.groupby(["SepetID", "Hizmet"])["Hizmet"].count().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)

frequent_itemsets = apriori(df_pivot,
                            min_support=0.01,
                            use_colnames=True,
                            low_memory=True)

frequent_itemsets.sort_values("support", ascending=False)

rules = association_rules(frequent_itemsets,
                          metric="support",
                          min_threshold=0.01)

sorted_rules = rules.sort_values("lift", ascending=False)

def arl_recommender(rules, product_id, rec_count=1):
    sorted_rules = rules.sort_values("lift", ascending=False)
    recommendation_list = []

    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]

arl_recommender(rules, "2_0", 5)