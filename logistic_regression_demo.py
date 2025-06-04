# ## Import Libraries
# import pandas as pd
# import re
# from mlxtend.frequent_patterns import association_rules, apriori
# from dash import Dash, dcc, html, Input, Output, callback
# import numpy as np
# import pickle
# import warnings
# warnings.filterwarnings('ignore')

# ## Load Dataset
# dataset_path = "dummy dataset.csv"  # Dataset path
# df = pd.read_csv(dataset_path)

# ## Preprocess text data
# df['lower_text'] = df['Text'].apply(lambda x: re.sub(r'\W+', ' ', str(x).lower()))  # Ensure text is string
# df.reset_index(inplace=True)

# ## Data Processing for Association Rule Mining
# full_data = pd.DataFrame()
# for i in df['index']:
#     words = df['lower_text'].iloc[i].split()
#     temp = pd.DataFrame({'item': words, 'index': i})
#     full_data = pd.concat([full_data, temp], axis=0)

# # Remove empty items and create pivot table
# transactions_str = full_data[full_data['item'] != ''].groupby(['index', 'item'])['item'].count().reset_index(name='Count')
# my_basket = transactions_str.pivot_table(index='index', columns='item', values='Count', aggfunc='sum').fillna(0)

# # Encode data as binary (0 or 1)
# def encode(x):
#     return 1 if x >= 1 else 0

# my_basket_sets = my_basket.applymap(encode)

# ## Association Rule Mining
# frequent_items = apriori(my_basket_sets, min_support=0.01, use_colnames=True)
# rules = association_rules(frequent_items, metric="lift", min_threshold=1)
# rules.sort_values('confidence', ascending=False, inplace=True)

# # Combine antecedents and consequents for predictions
# rules['antecedents_consequents'] = rules.apply(
#     lambda row: list(row['antecedents']) + list(row['consequents']), axis=1
# )

# # Save model
# with open('association_rules_model.pkl', 'wb') as file:
#     pickle.dump(rules, file)

# ## Create Dash Web Application
# external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

# # Load saved rules
# with open('association_rules_model.pkl', 'rb') as file:
#     saved_rules = pickle.load(file)

# app = Dash(__name__, external_stylesheets=external_stylesheets)

# app.layout = html.Div(
#     [
#         html.H1("Next Word Prediction using Apriori Algorithm"),
#         html.I("Enter your input here:"),
#         html.Br(),
#         dcc.Input(id="input1", type="text", placeholder="Sentence", style={'marginRight': '10px', 'width': '90%'}),
#         html.Br(), html.Br(),
#         html.Label("Predicted next words:"),
#         html.Div(id="output"),
#     ]
# )

# @callback(
#     Output("output", "children"),
#     Input("input1", "value"),
# )
# def update_output(input1):
#     if not input1:
#         return ""

#     new_antecedents = set(input1.lower().split())
    
#     # Find rules where antecedents include all input words
#     matched_rules = saved_rules[saved_rules['antecedents'].apply(lambda x: new_antecedents.issubset(x))]

#     # Extract recommendations not in input
#     if not matched_rules.empty:
#         matched_rules = matched_rules.sort_values('confidence', ascending=False).head(20)
#         matched_rules['Recommendations'] = matched_rules['antecedents_consequents'].apply(
#             lambda x: [word for word in x if word not in new_antecedents]
#         )
#         try:
#             recommended_words = set(np.concatenate(matched_rules['Recommendations'].to_numpy()))
#         except:
#             recommended_words = set()
#     else:
#         recommended_words = set()

#     return ', '.join(recommended_words) if recommended_words else "No suggestions found."

# if __name__ == "__main__":
#     app.run(debug=False)


## Import Libraries
import pandas as pd
import re
from mlxtend.frequent_patterns import association_rules, apriori
from dash import Dash, dcc, html, Input, Output, callback
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    df['lower_text'] = df['Text'].apply(lambda x: re.sub(r'\W+', ' ', str(x).lower()))
    df.reset_index(inplace=True)

    full_data = pd.DataFrame()
    for i in df['index']:
        words = df['lower_text'].iloc[i].split()
        temp = pd.DataFrame({'item': words, 'index': i})
        full_data = pd.concat([full_data, temp], axis=0)

    transactions = full_data[full_data['item'] != ''].groupby(['index', 'item'])['item'].count().reset_index(name='Count')
    basket = transactions.pivot_table(index='index', columns='item', values='Count', aggfunc='sum').fillna(0)
    basket_sets = basket.applymap(lambda x: 1 if x >= 1 else 0)
    return basket_sets
def train_and_save_rules(data, model_path='association_rules_model.pkl'):
    frequent_items = apriori(data, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_items, metric="lift", min_threshold=1)
    rules['antecedents_consequents'] = rules.apply(
        lambda row: list(row['antecedents']) + list(row['consequents']), axis=1)
    rules = rules.sort_values(by='confidence', ascending=False)
    with open(model_path, 'wb') as f:
        pickle.dump(rules, f)
def load_rules(model_path='association_rules_model.pkl'):
    with open(model_path, 'rb') as f:
        return pickle.load(f)
def get_recommendations(input_text, rules):
    tokens = [w for w in re.findall(r'\b[a-z]+\b', input_text.lower())]
    input_set = set(tokens)

    matched = rules[rules['antecedents'].apply(lambda x: input_set.issubset(x))]
    if matched.empty:
        return []
    matched = matched.sort_values(by=['confidence', 'lift'], ascending=False).head(10)
    matched['Recommendations'] = matched['antecedents_consequents'].apply(
        lambda x: [word for word in x if word not in input_set]
    )

    try:
        recommended_words = set(np.concatenate(matched['Recommendations'].to_numpy()))
    except:
        recommended_words = set()

    return list(recommended_words)

# ----------------- MAIN -----------------
# Run this block once to train & save model
# basket_sets = load_and_preprocess("dummy dataset.csv")
# train_and_save_rules(basket_sets)

rules = load_rules()
external_stylesheets = ["https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css"]
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Smart Word Predictor"
app.layout = html.Div([
    html.Div([
        html.H2("🧠 Smart Next Word Prediction", className="text-primary"),
        html.P("Enter a phrase, and get context-aware word suggestions using the Apriori algorithm."),
        dcc.Input(
            id="input1",
            type="text",
            placeholder="Type your sentence here...",
            style={'width': '100%', 'padding': '10px', 'marginBottom': '20px'}
        ),
        html.Div(id="output", className="alert alert-info", style={"fontWeight": "bold"}),
    ], className="container mt-4")
])
@callback(
    Output("output", "children"),
    Input("input1", "value")
)
def update_output(input1):
    if not input1:
        return "Suggestions will appear here..."
    suggestions = get_recommendations(input1, rules)
    return ", ".join(suggestions) if suggestions else "❌ No suggestions found."

if __name__ == "__main__":
    app.run(debug=False)
