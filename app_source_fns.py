import pandas as pd
# import tensorflow as tf
from keras.models import load_model
from datetime import datetime, timedelta
import os


def classify_storm_level(value):
    if value >= 7:
        return "Severe Storm"
    elif value >= 5:
        return "Possible Storm"
    else:
        return "Calm"
    

def get_previous_entries(date, dates_df, final_df):
    date = str(date)[:10]
    date = datetime.strftime(datetime.strptime(date, '%Y-%m-%d') - timedelta(days=1), format='%Y-%m-%d')
    # print(f"\nDate being used: {date}")
    filtered_dates = dates_df[dates_df == date]
    if not filtered_dates.empty:
        start_index = filtered_dates.index[0]
        # print(f"Starting index (for debugging): {start_index}")
        prev_entries = final_df.iloc[start_index - 59:start_index+1]
        # print(f"Previous Entries: {prev_entries}")
        return prev_entries
    else:
        print(f"Start date: {date} not found in the date DataFrame.")
        return None
    

def predict(date, dates_df, final_df, loaded_model):
    X_test = get_previous_entries(date, dates_df, final_df)
    # print(f"\nX_test shape: {X_test.shape}")
    X_test = X_test.values.reshape(1, 60, 1)
    Kp_predicted = loaded_model.predict(X_test)
    # print(f"\nKp_predicted: {Kp_predicted}")
    if date not in dates_df.values:
        # print("Appending additional values to the dates_df and final_df")
        # dates_df = dates_df._append(pd.Series([date[:10]]), ignore_index=True)
        dates_df = dates_df._append(pd.Series([date.strftime('%Y-%m-%d')[:10]]), ignore_index=True)
        final_df = final_df._append(pd.Series([Kp_predicted[0][0]]), ignore_index=True)
        # dates_df = pd.concat([dates_df, pd.Series([date])])
        # final_df = pd.concat([final_df, pd.Series([Kp_predicted[0][0]])])
        #print type of dates_df and final_df
        # print(f"Type of dates_df: {type(dates_df)}")
        # print(f"Type of final_df: {type(final_df)}")
        # print(dates_df.tail())
        # print(final_df.tail())

    return Kp_predicted[0][0], dates_df, final_df


def get_table(start_date, end_date, dates_df, final_df, loaded_model):
    predicted_Kp_values = []
    start_date = start_date.strftime('%Y-%m-%d')[:10]
    # print(f"\n\nStart date: {start_date}")
    end_date = end_date.strftime('%Y-%m-%d')[:10]
    # print(f"End date: {end_date}")  
    for date in pd.date_range(start=start_date, end=end_date):
        print(f"\n\n-----------------------------\nPredicting for date: {date}")
        predicted_Kp, dates_df2, final_df2 = predict(date, dates_df, final_df, loaded_model)
        dates_df = dates_df2
        final_df = final_df2
        storm_level = classify_storm_level(predicted_Kp)
        print(date, type(date))
        date1 = date.strftime('%Y-%m-%d')[:10]
        print(date1, type(date1))
        predicted_Kp_values.append((date1, predicted_Kp, storm_level))
    # pd.to_pickle(dates_df, "data/dates_df2")
    # pd.to_pickle(final_df, "data/final_df2")
    # print(dates_df.tail())
    # print(final_df.tail())
    return predicted_Kp_values, dates_df, final_df
    

def process(start_date, end_date):
    # Set current working directory as file's directory
    os.chdir(os.path.dirname(__file__))
    dates_df = pd.read_pickle("data/dates_df2").iloc[:, 4]
    final_df = pd.read_pickle("data/final_df").iloc[:,4]
    # loaded_model = tf.keras.models.load_model("chekpoint_models/1d_output_model")
    loaded_model = load_model("./chekpoint_models/1d_output_model_new.keras")

    # Prepare dates_df if needed
    last_date = dates_df.tail(1).item()
    last_date = datetime.strptime(last_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    if last_date < end_date:
        Kp_values, dates_df, final_df = get_table(last_date, end_date, dates_df, final_df, loaded_model)
        Kp2 = []
        for i, (date, kp, _class) in enumerate(Kp_values):
            if date == start_date:
                Kp2 = Kp_values[i:]
                break
        return Kp2


# start_date = '2023-01-02'
# end_date = '2023-01-04'
# predicted_Kp_values = process(start_date, end_date)
# print(f"\n\nPredicted Kp values: {predicted_Kp_values}")