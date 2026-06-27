from flask import Flask, render_template, request
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
PLOT_FOLDER = 'static/plots'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():

    file = request.files['file']

    if file.filename == '':
        return "No file selected"

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    df = pd.read_csv(filepath)

    # Select numeric columns only
    df = df.select_dtypes(include=['number'])

    # Remove missing values
    df = df.dropna()

    if len(df.columns) < 2:
        return "Need at least 2 numerical columns"

    # =========================
    # LINEAR REGRESSION
    # =========================

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)

    # Regression Plot
    plt.figure(figsize=(6,4))
    plt.scatter(y_test, predictions)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Linear Regression")

    regression_plot = 'plots/regression.png'
    plt.savefig(f'static/{regression_plot}')
    plt.close()

    # =========================
    # SCALING
    # =========================

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # =========================
    # KMEANS
    # =========================

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_labels = kmeans.fit_predict(scaled_data)

    plt.figure(figsize=(6,4))
    plt.scatter(
        scaled_data[:,0],
        scaled_data[:,1],
        c=kmeans_labels
    )

    plt.title("KMeans Clustering")

    kmeans_plot = 'plots/kmeans.png'
    plt.savefig(f'static/{kmeans_plot}')
    plt.close()

    # =========================
    # DBSCAN
    # =========================

    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(scaled_data)

    plt.figure(figsize=(6,4))
    plt.scatter(
        scaled_data[:,0],
        scaled_data[:,1],
        c=dbscan_labels
    )

    plt.title("DBSCAN Clustering")

    dbscan_plot = 'plots/dbscan.png'
    plt.savefig(f'static/{dbscan_plot}')
    plt.close()

    table = df.head().to_html(classes='table')

    return render_template(
        'index.html',
        table=table,
        r2=round(r2, 2),
        regression_plot=regression_plot,
        kmeans_plot=kmeans_plot,
        dbscan_plot=dbscan_plot
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
