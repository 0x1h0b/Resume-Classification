
import nltk
from nltk.corpus import stopwords
import string
import re
import joblib
import pandas as pd

import warnings
warnings.filterwarnings('ignore') 

nltk.data.path.append('./nltk_data/')


def clean_text(doc):
    ''' Pre-processing of document , basic cleaning and stopword removal '''
    doc = re.sub('http\S+\s*',' ',doc) # remove urls
    doc = re.sub('RT|cc',' ',doc) # RT and cc
    doc = re.sub('#\S+|@\S+',' ',doc) # hashtags and mentions
    doc = re.sub(r'[^\x00-\x7f]',' ', doc) 
    doc = re.sub('\s+',' ',doc) # remove extra spaces
    
    tokens = nltk.word_tokenize(doc)
    stopword_all =  list(set(stopwords.words('english'))) 
    stopword_all.extend(list(string.punctuation))
    cleaned_token = [token.lower() for token in tokens if token not in stopword_all]
    
    return ' '.join(cleaned_token)

def predict_class_mnb(text):
    cleaned_text = clean_text(text)

    # tfidf model
    tfidf_encoder = joblib.load('./models/naive_bayes/word_vec_encoder.pkl')
    text_vec = tfidf_encoder.transform([cleaned_text])

    # multinomial nb
    loaded_clf = joblib.load('./models/naive_bayes/multinomial_nb.pkl')
    ans = loaded_clf.predict(text_vec)
    # print(ans)

    #label encoder
    label_encoder = joblib.load('./models/naive_bayes/output_label_encoder.pkl')
    result = label_encoder.inverse_transform(ans)


    all_ans = loaded_clf.predict_proba(text_vec)[0]
    ans_list = []
    for idx,val in enumerate(all_ans):
        # print(idx)
        l=label_encoder.inverse_transform([idx])
        ans_list.append([val,l[0]])
    all_ans_df = pd.DataFrame(ans_list,columns=['val','label'])
    # print(all_ans_df)

    return result,all_ans_df




# x="himanshu bag himanshubag12 cid:211 +91 8295990851 www.linkedin.com/in/himanshu-bag/ cid:135 experience technical skills associate software engineer python c++ linux git qualcommindia datastructure oop cid:17 feb2021 present hyderabad india partofthemachinelearningceteamwhichworksonvariousml datascience numpy pandas matplotlib driventoolsusedinternallybyqualcommengineers machinelearning nlp basics workedwithrasaframeworktobuildcustomnermodeltoidentify di erentdomainspeci centitiesandintentsfromlog les deeplearning ann rnn lstm cnn projects education blog website content management system b.tech electronics php html css bootstrap jquery communication engineering cid:17 december2018 january2018 nitkurukshetra haryana basiclogin registerfunctionality userscanpostblogs cid:17 july2016 may2020 whileadmincanmanageposts users categories comments cgpa:7.4 blog website content management system senior secondary cbse board php html css bootstrap jquery jawaharnavodayavidyalaya puducherry cid:17 december2018 january2018 cid:17 2014 2016 basiclogin registerfunctionality userscanpostblogs percentage:89.40 marks 447/500 whileadmincanmanageposts users categories comments secondary cbse board jawaharnavodayavidyalaya nuapada blog website content management system cid:17 2014 php html css bootstrap jquery cgpa:9.6 cid:17 december2018 january2018 basiclogin registerfunctionality userscanpostblogs whileadmincanmanageposts users categories comments soft skills quicklearnerandcreative blog website content management system planningthenexecutionbelief php html css bootstrap jquery detailedandorganized cid:17 december2018 january2018 punctual basiclogin registerfunctionality userscanpostblogs whileadmincanmanageposts users categories comments hobbies ce ifications watching playingfootball drawingandsketching deeplearningspecialization-coursera competitiveprogramming introductiontodatasciencewithpython-coursera datasciencewithpython-datacamp competitiveprogramming-codingninja"


# ans,df = predict_class_mnb(x)
