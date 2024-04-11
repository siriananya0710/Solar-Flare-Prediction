import datetime
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from app_source_fns import *


def main():
    df = None
    start_date = None
    end_date = None
    st.title("Geomagnetic Storm Prediction tool using Kp Index")
    with st.sidebar:
        st.title("Configuration")
        start_date = str(st.date_input("Start date", value=None, key=1))
        end_date = str(st.date_input("End date", value=None, key=2))
        print("Processed dates")
        print(f"Start date received: {start_date}")
        print(f"End date received: {end_date}")
        if st.button("Predict Kp for Dates"):
            st.write("Predicting Kp index for the selected dates")
            predicted_Kp_table = process(start_date, end_date)
            df = pd.DataFrame(predicted_Kp_table, columns=['Date', 'Predicted Kp', 'Storm Level'])
    
    st.table(df)

if __name__ == "__main__":
    st.set_page_config(page_title="Solar Flare Detection")
    main()