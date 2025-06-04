import numpy as np 
import pandas as pd 
from mlxtend.frequent_patterns import apriori, association_rules 
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 🔹 Ma'lumotlarni yuklash
data = pd.read_csv('C:/Users/Sozla.uz/.vscode/.vscode/Online_Retail.csv')

# 🔹 Ma'lumotlarni tozalash
data['Description'] = data['Description'].str.strip()
data.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
data['InvoiceNo'] = data['InvoiceNo'].astype('str')
data = data[~data['InvoiceNo'].str.contains('C')]  # 'C' — qaytarilgan buyurtmalar

# 🔹 Tanlangan mamlakatlar
countries = ['France', 'United Kingdom', 'Portugal', 'Sweden']
baskets = {}

# 🔹 Har bir mamlakat uchun savat tayyorlash
for country in countries:
    baskets[country] = (data[data['Country'] == country]
                        .groupby(['InvoiceNo', 'Description'])['Quantity']
                        .sum().unstack().reset_index().fillna(0)
                        .set_index('InvoiceNo'))

# 🔹 1/0 binar holatga o'tkazish
def hot_encode(x):
    return 1 if x >= 1 else 0

for country in countries:
    baskets[country] = baskets[country].applymap(hot_encode)

# 🔹 Har bir mamlakat uchun assotsiatsiya qoidalarini chiqarish
rules_dict = {}

for country in countries:
    min_support = 0.05 if country != 'United Kingdom' else 0.01
    frq_items = apriori(baskets[country], min_support=min_support, use_colnames=True)
    rules = association_rules(frq_items, metric="lift", min_threshold=1)
    rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
    rules_dict[country] = rules
    print(f"\n📋 Rules for {country}:\n", rules.head(), "\n")

# 🔹 Chiroyli grafiklar chiqarish funksiyasi
def plot_rules(rules, country_name, top_n=10):
    plt.figure(figsize=(14, 10))
    sns.set(style="whitegrid", context='talk')  # Toza dizayn va katta shriftlar

    top_rules = rules.head(top_n)
    
    scatter = sns.scatterplot(
        x='support',
        y='confidence',
        size='lift',
        hue='lift',
        palette='coolwarm',  # yoki 'viridis', 'magma', 'plasma'
        sizes=(200, 1000),
        data=top_rules,
        alpha=0.8,
        edgecolor='black',
        linewidth=1.2
    )

    plt.title(f'💡 Association Rules in {country_name}', fontsize=22, weight='bold')
    plt.xlabel('🧮 Support', fontsize=16)
    plt.ylabel('📈 Confidence', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Referens chiziqlar
    plt.axhline(y=1, color='red', linestyle='--', label='Confidence = 1', linewidth=2)
    plt.axvline(x=0.05, color='green', linestyle='--', label='Support = 0.05', linewidth=2)

    # Legend
    handles, labels = scatter.get_legend_handles_labels()
    plt.legend(handles=handles[1:], labels=labels[1:], title="📊 Lift", fontsize=12, title_fontsize=13, loc='best', frameon=True)

    plt.grid(True, which='major', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

# 🔹 Har bir mamlakat uchun grafik chiqarish
for country in countries:
    plot_rules(rules_dict[country], country)

# 🔹 Joriy ishchi papkani ko‘rsatish
print("\n📁 Current Working Directory:", os.getcwd())
