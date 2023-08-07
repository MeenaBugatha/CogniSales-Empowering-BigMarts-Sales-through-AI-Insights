from flask import Flask, render_template, request, session
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle
from datetime import datetime
import os
import matplotlib.pyplot as plt
import joblib



app = Flask(__name__)
app.secret_key = 'meenabuga'

# Global variables
df = None
X_train = None
y_train = None
X_test = None
y_test = None
knn_model = None
rf_model = None
dt_model = None
encoder = None


def preprocess_data(df, encoder=None):
    non_numeric_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    if encoder is None:
        encoder = {}

    for col in non_numeric_cols:
        if df[col].dtype == 'object':
            if col not in encoder:
                encoder[col] = le.fit(df[col])
            df[col] = encoder[col].transform(df[col]) + 1

    df.fillna(0, inplace=True)

    return df, encoder


def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_model(filename):
    model = joblib.load(filename)
    return model


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/choose_dataset', methods=['POST'])
def choose_dataset():
    global df, X_train, y_train

    file = request.files['dataset']
    df = pd.read_csv(file)

    df, encoder = preprocess_data(df)

    X_train = df.drop(['Item_Outlet_Sales'], axis=1)
    y_train = df['Item_Outlet_Sales']

    return render_template('index.html', message='Dataset loaded successfully.')


@app.route('/split_dataset', methods=['POST'])
def split_dataset():
    global X_train, X_test, y_train, y_test

    try:
        test_size = None

        split_option = request.form['split_option']
        if split_option == '80_20':
            test_size = 0.2
        elif split_option == '60_40':
            test_size = 0.4
        elif split_option == 'custom':
            test_size = float(request.form['custom_split']) / 100
        elif split_option == 'k_fold':
            k = int(request.form['k_value'])
            kf = KFold(n_splits=k, shuffle=True, random_state=42)
            X_train_folds = []
            y_train_folds = []
            X_test_folds = []
            y_test_folds = []

            for train_index, test_index in kf.split(X_train):
                X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
                y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
                X_train_folds.append(X_train_fold)
                X_test_folds.append(X_test_fold)
                y_train_folds.append(y_train_fold)
                y_test_folds.append(y_test_fold)

            message = 'Dataset split using K-Fold successfully.'
            return render_template('index.html', message=message)

        else:
            raise ValueError('Invalid split option selected.')

        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_size, random_state=42)

        message = 'Dataset split successfully.'
        return render_template('index.html', message=message)

    except Exception as e:
        message = 'Error occurred during dataset split: {}'.format(str(e))
        return render_template('index.html', message=message)


@app.route('/select_algorithm', methods=['POST'])
def select_algorithm():
    global knn_model, rf_model, dt_model  # Add global declaration

    try:
        if X_test is None or y_test is None:
            raise ValueError('Test dataset not found. Please split the dataset first or load a pre-split dataset.')

        knn_model = KNeighborsRegressor()
        rf_model = RandomForestRegressor()
        dt_model = LinearRegression()

        knn_model.fit(X_train, y_train)
        rf_model.fit(X_train, y_train)
        dt_model.fit(X_train, y_train)

        knn_predictions = knn_model.predict(X_test)
        rf_predictions = rf_model.predict(X_test)
        dt_predictions = dt_model.predict(X_test)

        knn_mse = mean_squared_error(y_test, knn_predictions)
        rf_mse = mean_squared_error(y_test, rf_predictions)
        dt_mse = mean_squared_error(y_test, dt_predictions)

        now = datetime.now()
        current_time = now.strftime("%H-%M-%S--%d-%b-%Y")
        save_model(knn_model, 'knn_model-{}.pkl'.format(current_time))
        save_model(rf_model, 'random_forest_model-{}.pkl'.format(current_time))
        save_model(dt_model, 'linear_regression_model-{}.pkl'.format(current_time))

        selected_algorithm = request.form['algorithm']
        selected_mse = knn_mse if selected_algorithm == 'knn' else (rf_mse if selected_algorithm == 'random_forest' else dt_mse)

        # Store the selected algorithm in a session variable
        session['selected_algorithm'] = selected_algorithm

        return render_template('index.html', message='Models trained successfully.',
                           selected_algorithm=selected_algorithm, selected_mse=selected_mse)

    except Exception as e:
        return render_template('index.html', message='Error occurred during algorithm selection: {}'.format(str(e)))


@app.route('/select_pickle', methods=['POST'])
def select_pickle():

    try:
        pickle_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
        selected_pickle = request.form.get('pickle_selection')

        if not selected_pickle:  # Check if selected_pickle is empty or None
            return render_template('predict.html', pickle_files=pickle_files, selected_model=None, accuracy=None)

        if selected_pickle == 'current':
            selected_model = 'Current Model'
            accuracy = None
        else:
            selected_model = selected_pickle
            model = load_model(selected_pickle)
            accuracy = model.score(X_test, y_test)

        return render_template('predict.html', pickle_files=pickle_files, selected_model=selected_model,
                               accuracy=accuracy)

    except Exception as e:
        return render_template('predict.html', message='Error occurred during pickle selection: {}'.format(str(e)))


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global algorithm
    try:

        item_weight = float(request.form['item_weight'])
        item_visibility = float(request.form['item_visibility'])
        item_type = request.form['item_type']
        item_mrp = float(request.form['item_mrp'])
        outlet_establishment_year = int(request.form['outlet_establishment_year'])
        outlet_size = request.form['outlet_size']
        outlet_location_type = request.form['outlet_location_type']
        outlet_type = request.form['outlet_type']

        algorithm = session.get('selected_algorithm')  # Use 'selected_algorithm' instead of 'algorithm'

        # Prepare the input features for prediction
        input_features = pd.DataFrame({
            'Item_Weight': [item_weight],
            'Item_Visibility': [item_visibility],
            'Item_Type': [item_type],
            'Item_MRP': [item_mrp],
            'Outlet_Establishment_Year': [outlet_establishment_year],
            'Outlet_Size': [outlet_size],
            'Outlet_Location_Type': [outlet_location_type],
            'Outlet_Type': [outlet_type]
        })

        input_df = pd.DataFrame(input_features)
        input_df, encoder = preprocess_data(input_df, encoder)
        input_df = input_df.reindex(columns=X_train.columns, fill_value=0)

        if algorithm == 'knn':
            model = knn_model
        elif algorithm == 'random_forest':
            model = rf_model
        elif algorithm == 'linear_regression':
            model = dt_model
        else:
            return render_template('predict.html', message='Invalid algorithm selected.')

        prediction = model.predict(input_df)

        # Retrieve the target variable for display
        target_variable = y_train[0]  # Replace [0] with the appropriate index/column name if needed

        # You can format the prediction and target variable as needed
        formatted_prediction = prediction[0]  # Assuming prediction is a single value

        return render_template('predict.html', algorithm=algorithm, prediction=formatted_prediction,
                               target_variable=target_variable)

    except Exception as e:
        return render_template('predict.html', message='Error occurred during prediction: {}'.format(str(e)))


@app.route('/compare', methods=['POST'])
def compare():
    global knn_predictions, rf_predictions, dt_predictions  # Add global declaration
    knn_predictions = knn_model.predict(X_test)
    rf_predictions = rf_model.predict(X_test)
    dt_predictions = dt_model.predict(X_test)

    knn_mse = mean_squared_error(y_test, knn_predictions)
    rf_mse = mean_squared_error(y_test, rf_predictions)
    dt_mse = mean_squared_error(y_test, dt_predictions)

    models = ['KNN', 'Random Forest', 'Decision Tree']
    mse_values = [knn_mse, rf_mse, dt_mse]

    plt.bar(models, mse_values)
    plt.xlabel('Models')
    plt.ylabel('Mean Squared Error')
    plt.title('Model Mean Squared Error')
    plt.savefig('static/barchart.jpg')

    return render_template('result.html', knn_mse=knn_mse, rf_mse=rf_mse, dt_mse=dt_mse)


if __name__ == '__main__':
    app.run(debug=True)
