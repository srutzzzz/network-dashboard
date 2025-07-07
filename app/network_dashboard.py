import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff

# =====================
# LOAD + PREPROCESS
# =====================
df = pd.read_csv(r"C:\Users\LENOVO\Documents\Network log Analysis\data\UNSW_NB15_training-set.csv")


# clean up
df['attack_cat'] = df['attack_cat'].replace('-', 'Normal').fillna('Normal')

# anomaly detection with IsolationForest
features = ['dur', 'sbytes', 'dbytes', 'rate', 'smean', 'dmean']
iso = IsolationForest(contamination=0.02, random_state=42)
df['anomaly_flag'] = iso.fit_predict(df[features])
df['anomaly_flag'] = df['anomaly_flag'].map({1: 0, -1: 1})

# supervised ML: Random Forest
X = df[features]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
fig_ml = ff.create_annotated_heatmap(
    z=cm,
    x=["Normal","Attack"],
    y=["Normal","Attack"],
    colorscale="Blues",
    showscale=True
)
fig_ml.update_layout(title="Random Forest Confusion Matrix", height=350)

importances = clf.feature_importances_
feature_importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": importances
}).sort_values(by="importance", ascending=False)

fig_feature_importance = px.bar(
    feature_importance_df,
    x="importance",
    y="feature",
    orientation="h",
    color="importance",
    color_continuous_scale="blues",
    title="Random Forest Feature Importance"
)
fig_feature_importance.update_layout(height=350, margin=dict(t=20,b=20,l=20,r=20))

# =====================
# KPI metrics
# =====================
total_records = len(df)
total_attacks = df['label'].sum()
total_anomalies = df['anomaly_flag'].sum()

# =====================
# PLOTS
# =====================

# Normal vs Attack
normal_attack_counts = df['label'].value_counts()
fig1 = px.bar(
    x=['Normal', 'Attack'],
    y=normal_attack_counts.values,
    color=['Normal', 'Attack'],
    text=normal_attack_counts.values,
    color_discrete_sequence=["#3498DB", "#E74C3C"]
)
fig1.update_traces(textposition="outside")
fig1.update_layout(height=350, margin=dict(t=20,b=20,l=20,r=20), showlegend=False)

# Attack categories
attack_counts = df['attack_cat'].value_counts()
fig2 = px.bar(
    x=attack_counts.values,
    y=attack_counts.index,
    orientation="h",
    text=attack_counts.values,
    color=attack_counts.index,
    color_discrete_sequence=px.colors.qualitative.Safe
)
fig2.update_traces(textposition="outside")
fig2.update_layout(height=350, margin=dict(t=20,b=20,l=20,r=20), showlegend=False)

# Top protocols
proto_counts = df['proto'].value_counts()
proto_counts = proto_counts[proto_counts > 1000].head(10)
fig3 = px.bar(
    x=proto_counts.values,
    y=proto_counts.index,
    orientation="h",
    text=proto_counts.values,
    color=proto_counts.index,
    color_discrete_sequence=px.colors.qualitative.Pastel
)
fig3.update_traces(textposition="outside")
fig3.update_layout(height=350, margin=dict(t=20,b=20,l=20,r=20), showlegend=False)

# Top services
service_counts = df[df['service'] != '-']['service'].value_counts()
service_counts = service_counts[service_counts > 1000].head(10)
fig4 = px.bar(
    x=service_counts.index,
    y=service_counts.values,
    text=service_counts.values,
    color=service_counts.index,
    color_discrete_sequence=px.colors.qualitative.Pastel
)
fig4.update_traces(textposition="outside")
fig4.update_layout(height=350, margin=dict(t=20,b=20,l=20,r=20), showlegend=False)

# Numeric features histogram
numeric_features = ['dur', 'sbytes', 'dbytes']
fig5 = sp.make_subplots(rows=1, cols=3, subplot_titles=numeric_features)
for idx, feature in enumerate(numeric_features):
    fig5.add_trace(
        go.Histogram(x=df[feature], nbinsx=50, marker_color="#2980B9"),
        row=1, col=idx+1
    )
fig5.update_yaxes(type="log")
fig5.update_layout(height=350, margin=dict(t=20,b=20,l=20,r=20))

# Correlation heatmap
corr = df[features].corr()
fig6 = px.imshow(
    corr.round(2),
    text_auto=True,
    color_continuous_scale="blues"
)
fig6.update_layout(height=350, margin=dict(t=20,b=20,l=20,r=20))

# Anomalies by attack category
anomaly_counts = df[df['anomaly_flag']==1]['attack_cat'].value_counts()
fig7 = px.bar(
    x=anomaly_counts.index,
    y=anomaly_counts.values,
    text=anomaly_counts.values,
    color=anomaly_counts.index,
    color_discrete_sequence=px.colors.qualitative.Safe
)
fig7.update_traces(textposition="outside")
fig7.update_layout(
    title="Anomalies by Attack Category",
    height=350,
    margin=dict(t=20,b=20,l=20,r=20),
    showlegend=False
)

# Anomaly scatter
fig8 = px.scatter(
    df,
    x="sbytes",
    y="dbytes",
    color=df['anomaly_flag'].map({0:"Normal",1:"Anomaly"}),
    color_discrete_map={"Normal":"#3498DB", "Anomaly":"#E74C3C"},
    labels={"sbytes":"Source Bytes","dbytes":"Destination Bytes"},
    title="Anomaly Scatter Plot"
)
fig8.update_layout(height=350, margin=dict(t=20,b=20,l=20,r=20))

# Anomalies table
anomalies_table = dash_table.DataTable(
    data=df[df['anomaly_flag']==1][['proto','service','sbytes','dbytes']].head(10).to_dict('records'),
    columns=[{"name": i, "id": i} for i in ['proto','service','sbytes','dbytes']],
    style_table={'overflowX': 'auto'},
    style_cell={'textAlign': 'center'},
    style_header={'backgroundColor':'#3498DB','color':'white'}
)

# =====================
# DASH LAYOUT
# =====================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

app.layout = dbc.Container(fluid=True, children=[

    # header
    dbc.Row([
        dbc.Col(
            html.Div([
                html.H2("Network Traffic Analysis Dashboard", className="text-white"),
                html.Hr(),
                html.P("Network Traffic: Exploratory & Predictive Analysis", className="text-white-50"),
            ], className="p-4 bg-primary text-center shadow rounded"),
        )
    ], className="mb-4"),

    # KPI cards
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("Total Records"),
                html.H2(f"{total_records:,}", className="text-primary")
            ])
        ], className="shadow rounded text-center"), md=4),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("Total Attacks"),
                html.H2(f"{total_attacks:,}", className="text-danger")
            ])
        ], className="shadow rounded text-center"), md=4),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("Total Anomalies"),
                html.H2(f"{total_anomalies:,}", className="text-warning")
            ])
        ], className="shadow rounded text-center"), md=4),
    ], className="mb-4"),

    # first row
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Normal vs Attack"),
            dbc.CardBody(dcc.Graph(figure=fig1, config={"displayModeBar": False}))
        ], className="shadow rounded"), md=4),
        dbc.Col(dbc.Card([
            dbc.CardHeader("Attack Categories"),
            dbc.CardBody(dcc.Graph(figure=fig2, config={"displayModeBar": False}))
        ], className="shadow rounded"), md=4),
        dbc.Col(dbc.Card([
            dbc.CardHeader("Top Protocols"),
            dbc.CardBody(dcc.Graph(figure=fig3, config={"displayModeBar": False}))
        ], className="shadow rounded"), md=4),
    ], className="mb-4 mt-4"),

    # second row
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Top Services"),
            dbc.CardBody(dcc.Graph(figure=fig4, config={"displayModeBar": False}))
        ], className="shadow rounded"), md=4),
        dbc.Col(dbc.Card([
            dbc.CardHeader("Numeric Features Histogram"),
            dbc.CardBody(dcc.Graph(figure=fig5, config={"displayModeBar": False}))
        ], className="shadow rounded"), md=4),
        dbc.Col(dbc.Card([
            dbc.CardHeader("Correlation Heatmap"),
            dbc.CardBody(dcc.Graph(figure=fig6, config={"displayModeBar": False}))
        ], className="shadow rounded"), md=4),
    ], className="mb-4 mt-4"),

    # third row
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Anomalies by Attack Category"),
            dbc.CardBody(dcc.Graph(figure=fig7, config={"displayModeBar": False}))
        ], className="shadow rounded"), md=4),
        dbc.Col(dbc.Card([
            dbc.CardHeader("Anomaly Scatter Plot"),
            dbc.CardBody(dcc.Graph(figure=fig8, config={"displayModeBar": False}))
        ], className="shadow rounded"), md=4),
        dbc.Col(dbc.Card([
            dbc.CardHeader("Top 10 Anomalies Table"),
            dbc.CardBody(anomalies_table)
        ], className="shadow rounded"), md=4),
    ], className="mb-4 mt-4"),

    # final row for ML
    dbc.Row([
    dbc.Col(dbc.Card([
        dbc.CardHeader("Random Forest Confusion Matrix"),
        dbc.CardBody(dcc.Graph(figure=fig_ml, config={"displayModeBar": False}))
    ], className="shadow rounded"), md=6),

    dbc.Col(dbc.Card([
        dbc.CardHeader("Random Forest Feature Importance"),
        dbc.CardBody(dcc.Graph(figure=fig_feature_importance, config={"displayModeBar": False}))
    ], className="shadow rounded"), md=6),
], className="mb-4 mt-4")

])

if __name__ == "__main__":
    app.run(debug=True)