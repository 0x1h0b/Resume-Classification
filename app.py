import streamlit as st

import docx2txt
import pdfplumber
import traceback
import matplotlib.pyplot as plt
import numpy as np

from predict import *


def read_pdf(file):
    try:
        all_text = ""
        with pdfplumber.open(file) as pdf:
            pages = pdf.pages
            # st.write(pages)
            for page in pages:
                all_text += page.extract_text()
        return all_text
    except Exception:
        st.warning('Unable to read PDF file , refer to below error')
        st.error(traceback.print_exc())
        return ""


def process_text(text_data):

    text = clean_text(text_data)
    res,df = predict_class_mnb(text)
    st.markdown("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    st.subheader("Your Resume belongs to "+res[0]+" category !!")

    st.subheader("Below is the classification result ::- ")
    
    fig, ax = plt.subplots()
    y = df['val']
    x = df['label']
    
    # fig = plt.figure(figsize=(12, 5))
    plt.xticks(rotation=90)
    plt.ticklabel_format(style="plain")
    plt.bar(x,y)
    plt.xlabel('Labels')
    plt.ylabel("score")
    plt.title('Classification results')
    st.pyplot(fig)

    st.markdown("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    st.subheader("The cleaned text extracted from your resume ::-")
    st.write(text)


def main():
    try:
        st.title('Resume Classification')
        st.subheader('Input your Data')
        resume_text_area = st.text_area("Enter your Resume text data.")
        st.write('OR')
        resume_file_upload = st.file_uploader("Upload Your Resume !",type=["pdf","docx","txt"])

        if st.button('Submit'):    
            if resume_file_upload is not None:
                file_details = {
                    "filename":resume_file_upload.name,
                    "filetype":resume_file_upload.type,
                    "filesize":resume_file_upload.size
                }
                st.write(file_details)

                if resume_file_upload.type=="text/plain":
                    text = str(resume_file_upload.read(),"utf-8")
                    process_text(text)
                elif resume_file_upload.type=="application/pdf":
                    text = read_pdf(resume_file_upload)
                    process_text(text)
                else:
                    text = docx2txt.process(resume_file_upload)
                    process_text(text)
            elif resume_text_area is not None:
                process_text(resume_text_area)

        st.markdown("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)  
        col1, col2 = st.columns(2)
        col1.markdown(" Author :- Himanshu Bag")
        col2.markdown(" Github Link for this project : [Click-Me](https://github.com/0x1h0b/Resume-Classification)")
        col1, col2 = st.columns(2)
        col1.markdown(" LinkedIn Profile : [himanshu-bag](https://www.linkedin.com/in/himanshu-bag/)")
        col2.markdown(" Github Account : [0x1h0b](https://github.com/0x1h0b)")
    except Exception:
        st.write(traceback.print_exc())

if __name__=="__main__":
    main()