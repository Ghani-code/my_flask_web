from flask import Flask, render_template
import pandas as pd
import seaborn as sns
import zipfile
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

app = Flask(__name__, template_folder='Public')  # Update the template_folder parameter

@app.route('/')
def index():
    # Unzip the file
    with zipfile.ZipFile('Assignment-1_Data.zip', 'r') as zip_ref:
        zip_ref.extractall('.')

    # Read the CSV data
    data = pd.read_csv('Assignment-1_Data.csv', delimiter=';', dtype={'BillNo': str})
    data = data.dropna(axis=0, how='any')

    data['Itemname'] = data['Itemname'].str.strip()
    data['Itemname'] = data['Itemname'].str.lower()
    data['Country'] = data['Country'].str.strip()
    data['Country'] = data['Country'].str.lower()
    data['Date'] = pd.to_datetime(data['Date'], format="%d.%m.%Y %H:%M")

    data['Price'] = data['Price'].str.replace(',', '.')
    data['Price'] = pd.to_numeric(data['Price'])

    item_dict = {}
    itemID = 0

    for item in data['Itemname'].unique():
        itemID += 1
        item_dict[item] = itemID

    data['ItemID'] = data['Itemname'].map(item_dict)

    data['day'] = data['Date'].dt.day
    data['week_day'] = data['Date'].dt.dayofweek
    data['week_day'] = data['week_day'].replace((0, 1, 2, 3, 4, 5, 6),
                                                ('Senin', 'Selasa', 'Rabu', 'Kamis',
                                                 'Jumat', 'Sabtu', 'Minggu'))
    data['month'] = data['Date'].dt.month
    data['month'] = data['month'].replace((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
                                          ('Januari', 'Februari', 'Maret', 'April', 'Mei',
                                           'Juni', 'Juli', 'Agustus', 'September', 'Oktober',
                                           'November', 'Desember'))
    data['year'] = data['Date'].dt.year
    data['Date'] = data['Date'].dt.date

    orderday = data.groupby(data['day'])['BillNo'].count().reset_index()
    orderday.columns = ['day', 'BillNo']
    orderday.sort_values('day', inplace=True)

    orderweek = data.groupby(data['week_day'])['BillNo'].count().reset_index()
    orderweek['orderweek'] = [0, 1, 2, 3, 4, 5]
    orderweek.sort_values('week_day', ascending=False, inplace=True)

    ordermonth = data.groupby(data['month'])['BillNo'].count().reset_index()
    ordermonth.loc[:, 'ordermonth'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    ordermonth.sort_values('BillNo', ascending=False, inplace=True)

    orderyear = data.groupby(data['year'])['BillNo'].count().reset_index()
    orderyear.loc[:, 'orderyear'] = [2010, 2011]
    orderyear.sort_values('orderyear', inplace=True)

    ordercountry = data.groupby('Country')['BillNo'].count().reset_index(name='Count')
    ordercountry = ordercountry.sort_values('Count', ascending=False)
    ordercountry = ordercountry.head(10)

    ukitems = data[data['Country'] == 'united kingdom']

    aprioricount = data.groupby(['BillNo', 'Itemname'])['Itemname'].count().reset_index(name='Count')

    def encode(x):
        if x <= 0:
            return 0
        if x >= 1:
            return 1

    basket = aprioricount.pivot_table(index='BillNo', columns='Itemname',
                                      values='Count', aggfunc='sum').fillna(0)
    basket = basket.applymap(encode)

    frequent_items = apriori(basket, min_support=0.025, use_colnames=True)
    frequent_items.sort_values('support', ascending=False, inplace=True)

    rules = association_rules(frequent_items, metric="support", min_threshold=0.025)
    rules.sort_values('confidence', ascending=False, inplace=True)

    def recommend_products(item_id, association_rules):
        items = data.loc[data['ItemID'] == item_id, 'Itemname'].values
        recommended_items = []

        for antecedent in rules['antecedents']:
            if set(antecedent).issubset(items):
                consequent = association_rules.loc[association_rules['antecedents'] == antecedent, 'consequents']
                recommended_items.extend(consequent)

        return [', '.join(item) for item in recommended_items]


    recommended_items = recommend_products(67, rules)

    return render_template('index.html', orderday=orderday, orderweek=orderweek, ordermonth=ordermonth,
                           orderyear=orderyear, ordercountry=ordercountry, ukitems=ukitems,
                           recommended_items=recommended_items)

if __name__ == '__main__':
    app.run(debug=True)
