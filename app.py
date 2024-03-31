import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import gptparser as cp
import json

def main():
    load_dotenv('key.env')
    st.set_page_config(page_title="CV Extraction Bot")
    st.title("CV Extraction Bot...💁 ")
    st.subheader("I can help you in extracting CV data")

    cv_files = st.file_uploader("Upload CVs here, only PDF files allowed", type=["pdf"], accept_multiple_files=True)
    submit = st.button("Extract Data")

    if submit and cv_files:
        with st.spinner('Wait for it...'):
            json_data_list = json.loads(cp.create_docs(cv_files))  # Parse JSON string to a list of dictionaries
            st.write(json_data_list)

            # Convert the list of dictionaries to a DataFrame
            df = pd.DataFrame(json_data_list)

            # Display the DataFrame
            st.write(df.head())

            # Download button for CSV
            data_as_csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download data as CSV", 
                data_as_csv, 
                "cv_extraction_result.csv",
                "text/csv",
                key="download-cv-extraction-csv",
            )
        st.success("Data extraction completed!")

# Invoking main function
if __name__ == '__main__':
    main()
