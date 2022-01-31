import streamlit as st

import docx2txt
import pdfplumber
import traceback


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
    st.header('Here are the Results !')
    st.write(text_data)



def main():
    try:
        st.title('Resume Classification')
        st.header('Input your Data')
        resume_text_area = st.text_area("Enter your Resume text data.")
        st.subheader('OR')
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
        
    except Exception:
        st.write(traceback.print_exc())

if __name__=="__main__":
    main()