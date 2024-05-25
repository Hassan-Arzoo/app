# # # # # # # # # # # # # # # # # # # # # 

# # # # # # # # # # # # # # # # # # # # #To make the Streamlit app feel more like a chatbot, we can modify the layout and interaction style. Here's a revised version:


# # # # # # # # # # # # # # # # # # # # import streamlit as st
# # # # # # # # # # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # # # # # # # # # import pandas as pd
# # # # # # # # # # # # # # # # # # # # from sklearn.feature_extraction.text import TfidfVectorizer
# # # # # # # # # # # # # # # # # # # # from sklearn.naive_bayes import MultinomialNB
# # # # # # # # # # # # # # # # # # # # from sklearn.model_selection import train_test_split
# # # # # # # # # # # # # # # # # # # # import nltk
# # # # # # # # # # # # # # # # # # # # import re
# # # # # # # # # # # # # # # # # # # # from nltk.corpus import stopwords
# # # # # # # # # # # # # # # # # # # # import string
# # # # # # # # # # # # # # # # # # # # from wordcloud import WordCloud
# # # # # # # # # # # # # # # # # # # # import matplotlib.pyplot as plt

# # # # # # # # # # # # # # # # # # # # # Download stopwords
# # # # # # # # # # # # # # # # # # # # nltk.download('stopwords')

# # # # # # # # # # # # # # # # # # # # # Load data
# # # # # # # # # # # # # # # # # # # # df = pd.read_csv('stress.csv')

# # # # # # # # # # # # # # # # # # # # # Preprocess data
# # # # # # # # # # # # # # # # # # # # stemmer = nltk.SnowballStemmer("english")
# # # # # # # # # # # # # # # # # # # # stopword = set(stopwords.words('english'))

# # # # # # # # # # # # # # # # # # # # def clean(text):
# # # # # # # # # # # # # # # # # # # #     text = str(text).lower()
# # # # # # # # # # # # # # # # # # # #     text = re.sub(r'\[.*?\]', '', text)
# # # # # # # # # # # # # # # # # # # #     text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
# # # # # # # # # # # # # # # # # # # #     text = re.sub(r'\w*\d\w*', '', text)
# # # # # # # # # # # # # # # # # # # #     text = re.sub(r'[''""…]', '', text)
# # # # # # # # # # # # # # # # # # # #     text = re.sub(r'\n', '', text)
# # # # # # # # # # # # # # # # # # # #     text = [word for word in text.split(' ') if word not in stopword]
# # # # # # # # # # # # # # # # # # # #     text = ' '.join(text)
# # # # # # # # # # # # # # # # # # # #     text = [stemmer.stem(word) for word in text.split(' ')]
# # # # # # # # # # # # # # # # # # # #     text = ' '.join(text)
# # # # # # # # # # # # # # # # # # # #     return text

# # # # # # # # # # # # # # # # # # # # df['text'] = df['text'].apply(clean)

# # # # # # # # # # # # # # # # # # # # # Feature extraction
# # # # # # # # # # # # # # # # # # # # tfidf = TfidfVectorizer(max_features=5000)
# # # # # # # # # # # # # # # # # # # # X = tfidf.fit_transform(df['text']).toarray()
# # # # # # # # # # # # # # # # # # # # y = df['label']

# # # # # # # # # # # # # # # # # # # # # Train model
# # # # # # # # # # # # # # # # # # # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # # # # # # # # # # # # # # # # # # # model = MultinomialNB()
# # # # # # # # # # # # # # # # # # # # model.fit(X_train, y_train)

# # # # # # # # # # # # # # # # # # # # # Custom CSS for styling
# # # # # # # # # # # # # # # # # # # # st.markdown("""
# # # # # # # # # # # # # # # # # # # #     <style>
# # # # # # # # # # # # # # # # # # # #     .chat-container {
# # # # # # # # # # # # # # # # # # # #         background-color: #f5f5f5;
# # # # # # # # # # # # # # # # # # # #         padding: 2rem;
# # # # # # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # # # # # #     .message {
# # # # # # # # # # # # # # # # # # # #         background-color: #ffffff;
# # # # # # # # # # # # # # # # # # # #         padding: 1rem;
# # # # # # # # # # # # # # # # # # # #         border-radius: 10px;
# # # # # # # # # # # # # # # # # # # #         margin-bottom: 1rem;
# # # # # # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # # # # # #     .bot {
# # # # # # # # # # # # # # # # # # # #         text-align: left;
# # # # # # # # # # # # # # # # # # # #         color: #2c3e50;
# # # # # # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # # # # # #     .user {
# # # # # # # # # # # # # # # # # # # #         text-align: right;
# # # # # # # # # # # # # # # # # # # #         color: #2980b9;
# # # # # # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # # # # # #     .button {
# # # # # # # # # # # # # # # # # # # #         background-color: #3498db;
# # # # # # # # # # # # # # # # # # # #         color: white;
# # # # # # # # # # # # # # # # # # # #         border: none;
# # # # # # # # # # # # # # # # # # # #         padding: 1rem 2rem;
# # # # # # # # # # # # # # # # # # # #         font-size: 1.2rem;
# # # # # # # # # # # # # # # # # # # #         border-radius: 5px;
# # # # # # # # # # # # # # # # # # # #         cursor: pointer;
# # # # # # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # # # # # #     .button:hover {
# # # # # # # # # # # # # # # # # # # #         background-color: #2980b9;
# # # # # # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # # # # # #     </style>
# # # # # # # # # # # # # # # # # # # # """, unsafe_allow_html=True)

# # # # # # # # # # # # # # # # # # # # # Streamlit app layout
# # # # # # # # # # # # # # # # # # # # st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# # # # # # # # # # # # # # # # # # # # st.write("""
# # # # # # # # # # # # # # # # # # # # ### Stress Detection Chatbot
# # # # # # # # # # # # # # # # # # # # Hi there! I'm here to help you detect stress. Go ahead and type your message below.
# # # # # # # # # # # # # # # # # # # # """)

# # # # # # # # # # # # # # # # # # # # # Chat interaction
# # # # # # # # # # # # # # # # # # # # user_input = st.text_input("You:", "")
# # # # # # # # # # # # # # # # # # # # if st.button("Send"):
# # # # # # # # # # # # # # # # # # # #     if user_input:
# # # # # # # # # # # # # # # # # # # #         st.markdown('<div class="message user">' + user_input + '</div>', unsafe_allow_html=True)
# # # # # # # # # # # # # # # # # # # #         cleaned_input = clean(user_input)
# # # # # # # # # # # # # # # # # # # #         vectorized_input = tfidf.transform([cleaned_input]).toarray()
# # # # # # # # # # # # # # # # # # # #         prediction = model.predict(vectorized_input)
# # # # # # # # # # # # # # # # # # # #         result = "Stress Detected" if prediction[0] == 1 else "No Stress Detected"
# # # # # # # # # # # # # # # # # # # #         st.markdown('<div class="message bot">' + result + '</div>', unsafe_allow_html=True)

# # # # # # # # # # # # # # # # # # # # # Display word cloud
# # # # # # # # # # # # # # # # # # # # st.write("### Word Cloud of the Text Data")
# # # # # # # # # # # # # # # # # # # # wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['text']))
# # # # # # # # # # # # # # # # # # # # plt.figure(figsize=(10, 5))
# # # # # # # # # # # # # # # # # # # # plt.imshow(wordcloud, interpolation='bilinear')
# # # # # # # # # # # # # # # # # # # # plt.axis('off')
# # # # # # # # # # # # # # # # # # # # st.pyplot(plt)

# # # # # # # # # # # # # # # # # # # # # Add some information about the app
# # # # # # # # # # # # # # # # # # # # st.write("""
# # # # # # # # # # # # # # # # # # # # #### How it works:
# # # # # # # # # # # # # # # # # # # # - This chatbot uses a Naive Bayes classifier trained on textual data to detect stress.
# # # # # # # # # # # # # # # # # # # # - Type any message above and click on 'Send' to see if stress is detected.
# # # # # # # # # # # # # # # # # # # # - The model preprocesses the text by cleaning it and converting it into numerical features using TF-IDF vectorization.
# # # # # # # # # # # # # # # # # # # # """)

# # # # # # # # # # # # # # # # # # # # # Footer
# # # # # # # # # # # # # # # # # # # # st.markdown("""
# # # # # # # # # # # # # # # # # # # #     <div style='text-align: center; padding: 1rem; font-size: 0.9rem; color: #7f8c8d;'>
# # # # # # # # # # # # # # # # # # # #     Developed by [Your Name] - Final Year Project
# # # # # # # # # # # # # # # # # # # #     </div>
# # # # # # # # # # # # # # # # # # # # """, unsafe_allow_html=True)

# # # # # # # # # # # # # # # # # # # # st.markdown('</div>', unsafe_allow_html=True)


# # # # # # # # # # # # # # # # # # # # #This version transforms the app into a chat-style interface, where the user inputs messages and receives responses from the bot regarding stress detection.




# # # # # # # # # # # # # # # # # # # import streamlit as st
# # # # # # # # # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # # # # # # # # import pandas as pd
# # # # # # # # # # # # # # # # # # # from sklearn.feature_extraction.text import TfidfVectorizer
# # # # # # # # # # # # # # # # # # # from sklearn.naive_bayes import MultinomialNB
# # # # # # # # # # # # # # # # # # # from sklearn.model_selection import train_test_split
# # # # # # # # # # # # # # # # # # # import nltk
# # # # # # # # # # # # # # # # # # # import re
# # # # # # # # # # # # # # # # # # # from nltk.corpus import stopwords
# # # # # # # # # # # # # # # # # # # import string
# # # # # # # # # # # # # # # # # # # from wordcloud import WordCloud
# # # # # # # # # # # # # # # # # # # import matplotlib.pyplot as plt

# # # # # # # # # # # # # # # # # # # # Download stopwords
# # # # # # # # # # # # # # # # # # # nltk.download('stopwords')

# # # # # # # # # # # # # # # # # # # # Load data
# # # # # # # # # # # # # # # # # # # df = pd.read_csv('stress.csv')

# # # # # # # # # # # # # # # # # # # # Preprocess data
# # # # # # # # # # # # # # # # # # # stemmer = nltk.SnowballStemmer("english")
# # # # # # # # # # # # # # # # # # # stopword = set(stopwords.words('english'))

# # # # # # # # # # # # # # # # # # # def clean(text):
# # # # # # # # # # # # # # # # # # #     text = str(text).lower()
# # # # # # # # # # # # # # # # # # #     text = re.sub(r'\[.*?\]', '', text)
# # # # # # # # # # # # # # # # # # #     text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
# # # # # # # # # # # # # # # # # # #     text = re.sub(r'\w*\d\w*', '', text)
# # # # # # # # # # # # # # # # # # #     text = re.sub(r'[''""…]', '', text)
# # # # # # # # # # # # # # # # # # #     text = re.sub(r'\n', '', text)
# # # # # # # # # # # # # # # # # # #     text = [word for word in text.split(' ') if word not in stopword]
# # # # # # # # # # # # # # # # # # #     text = ' '.join(text)
# # # # # # # # # # # # # # # # # # #     text = [stemmer.stem(word) for word in text.split(' ')]
# # # # # # # # # # # # # # # # # # #     text = ' '.join(text)
# # # # # # # # # # # # # # # # # # #     return text

# # # # # # # # # # # # # # # # # # # df['text'] = df['text'].apply(clean)

# # # # # # # # # # # # # # # # # # # # Feature extraction
# # # # # # # # # # # # # # # # # # # tfidf = TfidfVectorizer(max_features=5000)
# # # # # # # # # # # # # # # # # # # X = tfidf.fit_transform(df['text']).toarray()
# # # # # # # # # # # # # # # # # # # y = df['label']

# # # # # # # # # # # # # # # # # # # # Train model
# # # # # # # # # # # # # # # # # # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # # # # # # # # # # # # # # # # # # model = MultinomialNB()
# # # # # # # # # # # # # # # # # # # model.fit(X_train, y_train)

# # # # # # # # # # # # # # # # # # # # Custom CSS for styling
# # # # # # # # # # # # # # # # # # # st.markdown("""
# # # # # # # # # # # # # # # # # # #     <style>
# # # # # # # # # # # # # # # # # # #     .chat-container {
# # # # # # # # # # # # # # # # # # #         display: flex;
# # # # # # # # # # # # # # # # # # #         flex-direction: row;
# # # # # # # # # # # # # # # # # # #         padding: 2rem;
# # # # # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # # # # #     .sidebar {
# # # # # # # # # # # # # # # # # # #         flex: 1;
# # # # # # # # # # # # # # # # # # #         padding: 0 2rem;
# # # # # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # # # # #     .main-content {
# # # # # # # # # # # # # # # # # # #         flex: 3;
# # # # # # # # # # # # # # # # # # #         padding: 0 2rem;
# # # # # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # # # # #     .message-container {
# # # # # # # # # # # # # # # # # # #         margin-bottom: 1rem;
# # # # # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # # # # #     .message {
# # # # # # # # # # # # # # # # # # #         background-color: #ffffff;
# # # # # # # # # # # # # # # # # # #         padding: 1rem;
# # # # # # # # # # # # # # # # # # #         border-radius: 10px;
# # # # # # # # # # # # # # # # # # #         margin-bottom: 0.5rem;
# # # # # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # # # # #     .bot {
# # # # # # # # # # # # # # # # # # #         text-align: left;
# # # # # # # # # # # # # # # # # # #         color: #2c3e50;
# # # # # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # # # # #     .user {
# # # # # # # # # # # # # # # # # # #         text-align: right;
# # # # # # # # # # # # # # # # # # #         color: #2980b9;
# # # # # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # # # # #     .button {
# # # # # # # # # # # # # # # # # # #         background-color: #3498db;
# # # # # # # # # # # # # # # # # # #         color: white;
# # # # # # # # # # # # # # # # # # #         border: none;
# # # # # # # # # # # # # # # # # # #         padding: 1rem 2rem;
# # # # # # # # # # # # # # # # # # #         font-size: 1.2rem;
# # # # # # # # # # # # # # # # # # #         border-radius: 5px;
# # # # # # # # # # # # # # # # # # #         cursor: pointer;
# # # # # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # # # # #     .button:hover {
# # # # # # # # # # # # # # # # # # #         background-color: #2980b9;
# # # # # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # # # # #     </style>
# # # # # # # # # # # # # # # # # # # """, unsafe_allow_html=True)

# # # # # # # # # # # # # # # # # # # # Streamlit app layout
# # # # # # # # # # # # # # # # # # # st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# # # # # # # # # # # # # # # # # # # # Sidebar for chat history
# # # # # # # # # # # # # # # # # # # st.sidebar.title("Chat History")
# # # # # # # # # # # # # # # # # # # chat_history = st.sidebar.empty()

# # # # # # # # # # # # # # # # # # # st.write("""
# # # # # # # # # # # # # # # # # # # ### Stress Detection Chatbot
# # # # # # # # # # # # # # # # # # # Hi there! I'm here to help you detect stress. Go ahead and type your message below.
# # # # # # # # # # # # # # # # # # # """)

# # # # # # # # # # # # # # # # # # # # Chat interaction
# # # # # # # # # # # # # # # # # # # user_input = st.text_input("You:", "")
# # # # # # # # # # # # # # # # # # # if st.button("Send"):
# # # # # # # # # # # # # # # # # # #     if user_input:
# # # # # # # # # # # # # # # # # # #         st.markdown('<div class="message-container">', unsafe_allow_html=True)
# # # # # # # # # # # # # # # # # # #         st.markdown('<div class="message user">' + user_input + '</div>', unsafe_allow_html=True)
# # # # # # # # # # # # # # # # # # #         cleaned_input = clean(user_input)
# # # # # # # # # # # # # # # # # # #         vectorized_input = tfidf.transform([cleaned_input]).toarray()
# # # # # # # # # # # # # # # # # # #         prediction = model.predict(vectorized_input)
# # # # # # # # # # # # # # # # # # #         result = "Stress Detected" if prediction[0] == 1 else "No Stress Detected"
# # # # # # # # # # # # # # # # # # #         st.markdown('<div class="message bot">' + result + '</div>', unsafe_allow_html=True)
# # # # # # # # # # # # # # # # # # #         st.markdown('</div>', unsafe_allow_html=True)

# # # # # # # # # # # # # # # # # # #         # Update chat history
# # # # # # # # # # # # # # # # # # #         chat_history.markdown('<div class="message-container">', unsafe_allow_html=True)
# # # # # # # # # # # # # # # # # # #         chat_history.markdown('<div class="message user">' + user_input + '</div>', unsafe_allow_html=True)
# # # # # # # # # # # # # # # # # # #         chat_history.markdown('<div class="message bot">' + result + '</div>', unsafe_allow_html=True)
# # # # # # # # # # # # # # # # # # #         chat_history.markdown('</div>', unsafe_allow_html=True)

# # # # # # # # # # # # # # # # # # # # Display word cloud
# # # # # # # # # # # # # # # # # # # st.write("### Word Cloud of the Text Data")
# # # # # # # # # # # # # # # # # # # wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['text']))
# # # # # # # # # # # # # # # # # # # plt.figure(figsize=(10, 5))
# # # # # # # # # # # # # # # # # # # plt.imshow(wordcloud, interpolation='bilinear')
# # # # # # # # # # # # # # # # # # # plt.axis('off')
# # # # # # # # # # # # # # # # # # # st.pyplot(plt)

# # # # # # # # # # # # # # # # # # # # Add some information about the app
# # # # # # # # # # # # # # # # # # # st.write("""
# # # # # # # # # # # # # # # # # # # #### How it works:
# # # # # # # # # # # # # # # # # # # - This chatbot uses a Naive Bayes classifier trained on textual data to detect stress.
# # # # # # # # # # # # # # # # # # # - Type any message above and click on 'Send' to see if stress is detected.
# # # # # # # # # # # # # # # # # # # - The model preprocesses the text by cleaning it and converting it into numerical features using TF-IDF vectorization.
# # # # # # # # # # # # # # # # # # # """)

# # # # # # # # # # # # # # # # # # # # Footer
# # # # # # # # # # # # # # # # # # # st.markdown("""
# # # # # # # # # # # # # # # # # # #     <div style='text-align: center; padding: 1rem; font-size: 0.9rem; color: #7f8c8d;'>
# # # # # # # # # # # # # # # # # # #     Developed by [Your Name] - Final Year Project
# # # # # # # # # # # # # # # # # # #     </div>
# # # # # # # # # # # # # # # # # # # """, unsafe_allow_html=True)

# # # # # # # # # # # # # # # # # # # st.markdown('</div>', unsafe_allow_html=True)




# # # # # # # # # # # # # # # # # # import streamlit as st
# # # # # # # # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # # # # # # # import pandas as pd
# # # # # # # # # # # # # # # # # # from sklearn.feature_extraction.text import TfidfVectorizer
# # # # # # # # # # # # # # # # # # from sklearn.naive_bayes import MultinomialNB
# # # # # # # # # # # # # # # # # # from sklearn.model_selection import train_test_split
# # # # # # # # # # # # # # # # # # import nltk
# # # # # # # # # # # # # # # # # # import re
# # # # # # # # # # # # # # # # # # from nltk.corpus import stopwords
# # # # # # # # # # # # # # # # # # import string
# # # # # # # # # # # # # # # # # # from wordcloud import WordCloud
# # # # # # # # # # # # # # # # # # import matplotlib.pyplot as plt

# # # # # # # # # # # # # # # # # # # Download stopwords
# # # # # # # # # # # # # # # # # # nltk.download('stopwords')

# # # # # # # # # # # # # # # # # # # Load data
# # # # # # # # # # # # # # # # # # df = pd.read_csv('stress.csv')

# # # # # # # # # # # # # # # # # # # Preprocess data
# # # # # # # # # # # # # # # # # # stemmer = nltk.SnowballStemmer("english")
# # # # # # # # # # # # # # # # # # stopword = set(stopwords.words('english'))

# # # # # # # # # # # # # # # # # # def clean(text):
# # # # # # # # # # # # # # # # # #     text = str(text).lower()
# # # # # # # # # # # # # # # # # #     text = re.sub(r'\[.*?\]', '', text)
# # # # # # # # # # # # # # # # # #     text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
# # # # # # # # # # # # # # # # # #     text = re.sub(r'\w*\d\w*', '', text)
# # # # # # # # # # # # # # # # # #     text = re.sub(r'[''""…]', '', text)
# # # # # # # # # # # # # # # # # #     text = re.sub(r'\n', '', text)
# # # # # # # # # # # # # # # # # #     text = [word for word in text.split(' ') if word not in stopword]
# # # # # # # # # # # # # # # # # #     text = ' '.join(text)
# # # # # # # # # # # # # # # # # #     text = [stemmer.stem(word) for word in text.split(' ')]
# # # # # # # # # # # # # # # # # #     text = ' '.join(text)
# # # # # # # # # # # # # # # # # #     return text

# # # # # # # # # # # # # # # # # # df['text'] = df['text'].apply(clean)

# # # # # # # # # # # # # # # # # # # Feature extraction
# # # # # # # # # # # # # # # # # # tfidf = TfidfVectorizer(max_features=5000)
# # # # # # # # # # # # # # # # # # X = tfidf.fit_transform(df['text']).toarray()
# # # # # # # # # # # # # # # # # # y = df['label']

# # # # # # # # # # # # # # # # # # # Train model
# # # # # # # # # # # # # # # # # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # # # # # # # # # # # # # # # # # model = MultinomialNB()
# # # # # # # # # # # # # # # # # # model.fit(X_train, y_train)

# # # # # # # # # # # # # # # # # # # Custom CSS for styling
# # # # # # # # # # # # # # # # # # st.markdown("""
# # # # # # # # # # # # # # # # # #     <style>
# # # # # # # # # # # # # # # # # #     .chat-container {
# # # # # # # # # # # # # # # # # #         display: flex;
# # # # # # # # # # # # # # # # # #         flex-direction: row;
# # # # # # # # # # # # # # # # # #         padding: 2rem;
# # # # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # # # #     .sidebar {
# # # # # # # # # # # # # # # # # #         flex: 1;
# # # # # # # # # # # # # # # # # #         padding: 0 2rem;
# # # # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # # # #     .main-content {
# # # # # # # # # # # # # # # # # #         flex: 3;
# # # # # # # # # # # # # # # # # #         padding: 0 2rem;
# # # # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # # # #     .message-container {
# # # # # # # # # # # # # # # # # #         margin-bottom: 1rem;
# # # # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # # # #     .message {
# # # # # # # # # # # # # # # # # #         background-color: #ffffff;
# # # # # # # # # # # # # # # # # #         padding: 1rem;
# # # # # # # # # # # # # # # # # #         border-radius: 10px;
# # # # # # # # # # # # # # # # # #         margin-bottom: 0.5rem;
# # # # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # # # #     .bot {
# # # # # # # # # # # # # # # # # #         text-align: left;
# # # # # # # # # # # # # # # # # #         color: #2c3e50;
# # # # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # # # #     .user {
# # # # # # # # # # # # # # # # # #         text-align: right;
# # # # # # # # # # # # # # # # # #         color: #2980b9;
# # # # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # # # #     .button {
# # # # # # # # # # # # # # # # # #         background-color: #3498db;
# # # # # # # # # # # # # # # # # #         color: white;
# # # # # # # # # # # # # # # # # #         border: none;
# # # # # # # # # # # # # # # # # #         padding: 1rem 2rem;
# # # # # # # # # # # # # # # # # #         font-size: 1.2rem;
# # # # # # # # # # # # # # # # # #         border-radius: 5px;
# # # # # # # # # # # # # # # # # #         cursor: pointer;
# # # # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # # # #     .button:hover {
# # # # # # # # # # # # # # # # # #         background-color: #2980b9;
# # # # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # # # #     </style>
# # # # # # # # # # # # # # # # # # """, unsafe_allow_html=True)

# # # # # # # # # # # # # # # # # # # Streamlit app layout
# # # # # # # # # # # # # # # # # # st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# # # # # # # # # # # # # # # # # # # Sidebar for chat history
# # # # # # # # # # # # # # # # # # st.sidebar.title("Chat History")
# # # # # # # # # # # # # # # # # # chat_history = st.sidebar.empty()

# # # # # # # # # # # # # # # # # # st.write("""
# # # # # # # # # # # # # # # # # # ### Stress Detection Chatbot
# # # # # # # # # # # # # # # # # # Hi there! I'm here to help you detect stress. Go ahead and type your message below.
# # # # # # # # # # # # # # # # # # """)

# # # # # # # # # # # # # # # # # # # Initialize chat history
# # # # # # # # # # # # # # # # # # chat_history_list = []

# # # # # # # # # # # # # # # # # # # Chat interaction
# # # # # # # # # # # # # # # # # # user_input = st.text_input("You:", "")
# # # # # # # # # # # # # # # # # # if st.button("Send"):
# # # # # # # # # # # # # # # # # #     if user_input:
# # # # # # # # # # # # # # # # # #         st.markdown('<div class="message-container">', unsafe_allow_html=True)
# # # # # # # # # # # # # # # # # #         st.markdown('<div class="message user">' + user_input + '</div>', unsafe_allow_html=True)
# # # # # # # # # # # # # # # # # #         cleaned_input = clean(user_input)
# # # # # # # # # # # # # # # # # #         vectorized_input = tfidf.transform([cleaned_input]).toarray()
# # # # # # # # # # # # # # # # # #         prediction = model.predict(vectorized_input)
# # # # # # # # # # # # # # # # # #         result = "Stress Detected" if prediction[0] == 1 else "No Stress Detected"
# # # # # # # # # # # # # # # # # #         st.markdown('<div class="message bot">' + result + '</div>', unsafe_allow_html=True)
# # # # # # # # # # # # # # # # # #         st.markdown('</div>', unsafe_allow_html=True)

# # # # # # # # # # # # # # # # # #         # Update chat history
# # # # # # # # # # # # # # # # # #         chat_history_list.append({'user': user_input, 'bot': result})
# # # # # # # # # # # # # # # # # #         chat_history.write('')
# # # # # # # # # # # # # # # # # #         for message in chat_history_list:
# # # # # # # # # # # # # # # # # #             chat_history.markdown('<div class="message-container">', unsafe_allow_html=True)
# # # # # # # # # # # # # # # # # #             chat_history.markdown('<div class="message user">' + message['user'] + '</div>', unsafe_allow_html=True)
# # # # # # # # # # # # # # # # # #             chat_history.markdown('<div class="message bot">' + message['bot'] + '</div>', unsafe_allow_html=True)
# # # # # # # # # # # # # # # # # #             chat_history.markdown('</div>', unsafe_allow_html=True)

# # # # # # # # # # # # # # # # # # # Display word cloud
# # # # # # # # # # # # # # # # # # st.write("### Word Cloud of the Text Data")
# # # # # # # # # # # # # # # # # # wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['text']))
# # # # # # # # # # # # # # # # # # plt.figure(figsize=(10, 5))
# # # # # # # # # # # # # # # # # # plt.imshow(wordcloud, interpolation='bilinear')
# # # # # # # # # # # # # # # # # # plt.axis('off')
# # # # # # # # # # # # # # # # # # st.pyplot(plt)

# # # # # # # # # # # # # # # # # # # Add some information about the app
# # # # # # # # # # # # # # # # # # st.write("""
# # # # # # # # # # # # # # # # # # #### How it works:
# # # # # # # # # # # # # # # # # # - This chatbot uses a Naive Bayes classifier trained on textual data to detect stress.
# # # # # # # # # # # # # # # # # # - Type any message above and click on 'Send' to see if stress is detected.
# # # # # # # # # # # # # # # # # # - The model preprocesses the text by cleaning it and converting it into numerical features using TF-IDF vectorization.
# # # # # # # # # # # # # # # # # # """)

# # # # # # # # # # # # # # # # # # # Footer
# # # # # # # # # # # # # # # # # # st.markdown("""
# # # # # # # # # # # # # # # # # #     <div style='text-align: center; padding: 1rem; font-size: 0.9rem; color: #7f8c8d;'>
# # # # # # # # # # # # # # # # # #     Developed by [Your Name] - Final Year Project
# # # # # # # # # # # # # # # # # #     </div>
# # # # # # # # # # # # # # # # # # """, unsafe_allow_html=True)

# # # # # # # # # # # # # # # # # # st.markdown('</div>', unsafe_allow_html=True)


# # # # # # # # # # # # # # # # # import streamlit as st
# # # # # # # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # # # # # # import pandas as pd
# # # # # # # # # # # # # # # # # from sklearn.feature_extraction.text import TfidfVectorizer
# # # # # # # # # # # # # # # # # from sklearn.naive_bayes import MultinomialNB
# # # # # # # # # # # # # # # # # from sklearn.model_selection import train_test_split
# # # # # # # # # # # # # # # # # import nltk
# # # # # # # # # # # # # # # # # import re
# # # # # # # # # # # # # # # # # from nltk.corpus import stopwords
# # # # # # # # # # # # # # # # # import string
# # # # # # # # # # # # # # # # # from wordcloud import WordCloud
# # # # # # # # # # # # # # # # # import matplotlib.pyplot as plt

# # # # # # # # # # # # # # # # # # Download stopwords
# # # # # # # # # # # # # # # # # nltk.download('stopwords')

# # # # # # # # # # # # # # # # # # Load data
# # # # # # # # # # # # # # # # # df = pd.read_csv('stress.csv')

# # # # # # # # # # # # # # # # # # Preprocess data
# # # # # # # # # # # # # # # # # stemmer = nltk.SnowballStemmer("english")
# # # # # # # # # # # # # # # # # stopword = set(stopwords.words('english'))

# # # # # # # # # # # # # # # # # def clean(text):
# # # # # # # # # # # # # # # # #     text = str(text).lower()
# # # # # # # # # # # # # # # # #     text = re.sub(r'\[.*?\]', '', text)
# # # # # # # # # # # # # # # # #     text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
# # # # # # # # # # # # # # # # #     text = re.sub(r'\w*\d\w*', '', text)
# # # # # # # # # # # # # # # # #     text = re.sub(r'[''""…]', '', text)
# # # # # # # # # # # # # # # # #     text = re.sub(r'\n', '', text)
# # # # # # # # # # # # # # # # #     text = [word for word in text.split(' ') if word not in stopword]
# # # # # # # # # # # # # # # # #     text = ' '.join(text)
# # # # # # # # # # # # # # # # #     text = [stemmer.stem(word) for word in text.split(' ')]
# # # # # # # # # # # # # # # # #     text = ' '.join(text)
# # # # # # # # # # # # # # # # #     return text

# # # # # # # # # # # # # # # # # df['text'] = df['text'].apply(clean)

# # # # # # # # # # # # # # # # # # Feature extraction
# # # # # # # # # # # # # # # # # tfidf = TfidfVectorizer(max_features=5000)
# # # # # # # # # # # # # # # # # X = tfidf.fit_transform(df['text']).toarray()
# # # # # # # # # # # # # # # # # y = df['label']

# # # # # # # # # # # # # # # # # # Train model
# # # # # # # # # # # # # # # # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # # # # # # # # # # # # # # # # model = MultinomialNB()
# # # # # # # # # # # # # # # # # model.fit(X_train, y_train)

# # # # # # # # # # # # # # # # # # Custom CSS for styling
# # # # # # # # # # # # # # # # # st.markdown("""
# # # # # # # # # # # # # # # # #     <style>
# # # # # # # # # # # # # # # # #     .chat-container {
# # # # # # # # # # # # # # # # #         display: flex;
# # # # # # # # # # # # # # # # #         flex-direction: row;
# # # # # # # # # # # # # # # # #         padding: 2rem;
# # # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # # #     .sidebar {
# # # # # # # # # # # # # # # # #         flex: 1;
# # # # # # # # # # # # # # # # #         padding: 0 2rem;
# # # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # # #     .main-content {
# # # # # # # # # # # # # # # # #         flex: 3;
# # # # # # # # # # # # # # # # #         padding: 0 2rem;
# # # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # # #     .message-container {
# # # # # # # # # # # # # # # # #         margin-bottom: 1rem;
# # # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # # #     .message {
# # # # # # # # # # # # # # # # #         background-color: #ffffff;
# # # # # # # # # # # # # # # # #         padding: 1rem;
# # # # # # # # # # # # # # # # #         border-radius: 10px;
# # # # # # # # # # # # # # # # #         margin-bottom: 0.5rem;
# # # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # # #     .bot {
# # # # # # # # # # # # # # # # #         text-align: left;
# # # # # # # # # # # # # # # # #         color: #2c3e50;
# # # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # # #     .user {
# # # # # # # # # # # # # # # # #         text-align: right;
# # # # # # # # # # # # # # # # #         color: #2980b9;
# # # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # # #     .button {
# # # # # # # # # # # # # # # # #         background-color: #3498db;
# # # # # # # # # # # # # # # # #         color: white;
# # # # # # # # # # # # # # # # #         border: none;
# # # # # # # # # # # # # # # # #         padding: 1rem 2rem;
# # # # # # # # # # # # # # # # #         font-size: 1.2rem;
# # # # # # # # # # # # # # # # #         border-radius: 5px;
# # # # # # # # # # # # # # # # #         cursor: pointer;
# # # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # # #     .button:hover {
# # # # # # # # # # # # # # # # #         background-color: #2980b9;
# # # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # # #     </style>
# # # # # # # # # # # # # # # # # """, unsafe_allow_html=True)

# # # # # # # # # # # # # # # # # # Streamlit app layout
# # # # # # # # # # # # # # # # # st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# # # # # # # # # # # # # # # # # # Sidebar for chat history
# # # # # # # # # # # # # # # # # st.sidebar.title("Chat History")
# # # # # # # # # # # # # # # # # chat_history = st.sidebar.empty()

# # # # # # # # # # # # # # # # # st.write("""
# # # # # # # # # # # # # # # # # ### Stress Detection Chatbot
# # # # # # # # # # # # # # # # # Hi there! I'm here to help you detect stress. Go ahead and type your message below.
# # # # # # # # # # # # # # # # # """)

# # # # # # # # # # # # # # # # # # Initialize chat history
# # # # # # # # # # # # # # # # # chat_history_list = []

# # # # # # # # # # # # # # # # # # Chat interaction
# # # # # # # # # # # # # # # # # user_input = st.text_input("You:", "")
# # # # # # # # # # # # # # # # # if st.button("Send"):
# # # # # # # # # # # # # # # # #     if user_input:
# # # # # # # # # # # # # # # # #         st.markdown('<div class="message-container">', unsafe_allow_html=True)
# # # # # # # # # # # # # # # # #         st.markdown('<div class="message user">' + user_input + '</div>', unsafe_allow_html=True)
# # # # # # # # # # # # # # # # #         cleaned_input = clean(user_input)
# # # # # # # # # # # # # # # # #         vectorized_input = tfidf.transform([cleaned_input]).toarray()
# # # # # # # # # # # # # # # # #         prediction = model.predict(vectorized_input)
# # # # # # # # # # # # # # # # #         result = "Stress Detected" if prediction[0] == 1 else "No Stress Detected"
# # # # # # # # # # # # # # # # #         st.markdown('<div class="message bot">' + result + '</div>', unsafe_allow_html=True)
# # # # # # # # # # # # # # # # #         st.markdown('</div>', unsafe_allow_html=True)

# # # # # # # # # # # # # # # # #         # Update chat history
# # # # # # # # # # # # # # # # #         chat_history_list.append({'user': user_input, 'bot': result})
# # # # # # # # # # # # # # # # #         chat_history.markdown('')
# # # # # # # # # # # # # # # # #         for message in chat_history_list:
# # # # # # # # # # # # # # # # #             chat_history.markdown('<div class="message-container">', unsafe_allow_html=True)
# # # # # # # # # # # # # # # # #             chat_history.markdown('<div class="message user">' + message['user'] + '</div>', unsafe_allow_html=True)
# # # # # # # # # # # # # # # # #             chat_history.markdown('<div class="message bot">' + message['bot'] + '</div>', unsafe_allow_html=True)
# # # # # # # # # # # # # # # # #             chat_history.markdown('</div>', unsafe_allow_html=True)

# # # # # # # # # # # # # # # # # # Display word cloud
# # # # # # # # # # # # # # # # # st.write("### Word Cloud of the Text Data")
# # # # # # # # # # # # # # # # # wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['text']))
# # # # # # # # # # # # # # # # # plt.figure(figsize=(10, 5))
# # # # # # # # # # # # # # # # # plt.imshow(wordcloud, interpolation='bilinear')
# # # # # # # # # # # # # # # # # plt.axis('off')
# # # # # # # # # # # # # # # # # st.pyplot(plt)

# # # # # # # # # # # # # # # # # # Add some information about the app
# # # # # # # # # # # # # # # # # st.write("""
# # # # # # # # # # # # # # # # # #### How it works:
# # # # # # # # # # # # # # # # # - This chatbot uses a Naive Bayes classifier trained on textual data to detect stress.
# # # # # # # # # # # # # # # # # - Type any message above and click on 'Send' to see if stress is detected.
# # # # # # # # # # # # # # # # # - The model preprocesses the text by cleaning it and converting it into numerical features using TF-IDF vectorization.
# # # # # # # # # # # # # # # # # """)

# # # # # # # # # # # # # # # # # # Footer
# # # # # # # # # # # # # # # # # st.markdown("""
# # # # # # # # # # # # # # # # #     <div style='text-align: center; padding: 1rem; font-size: 0.9rem; color: #7f8c8d;'>
# # # # # # # # # # # # # # # # #     Developed by [Your Name] - Final Year Project
# # # # # # # # # # # # # # # # #     </div>
# # # # # # # # # # # # # # # # # """, unsafe_allow_html=True)

# # # # # # # # # # # # # # # # # st.markdown('</div>', unsafe_allow_html=True)



# # # # # # # # # # # # # # # # import streamlit as st
# # # # # # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # # # # # import pandas as pd
# # # # # # # # # # # # # # # # from sklearn.feature_extraction.text import TfidfVectorizer
# # # # # # # # # # # # # # # # from sklearn.naive_bayes import MultinomialNB
# # # # # # # # # # # # # # # # from sklearn.model_selection import train_test_split
# # # # # # # # # # # # # # # # import nltk
# # # # # # # # # # # # # # # # import re
# # # # # # # # # # # # # # # # from nltk.corpus import stopwords
# # # # # # # # # # # # # # # # import string
# # # # # # # # # # # # # # # # from wordcloud import WordCloud
# # # # # # # # # # # # # # # # import matplotlib.pyplot as plt
# # # # # # # # # # # # # # # # from streamlit import caching

# # # # # # # # # # # # # # # # # Download stopwords
# # # # # # # # # # # # # # # # nltk.download('stopwords')

# # # # # # # # # # # # # # # # # Load data
# # # # # # # # # # # # # # # # df = pd.read_csv('stress.csv')

# # # # # # # # # # # # # # # # # Preprocess data
# # # # # # # # # # # # # # # # stemmer = nltk.SnowballStemmer("english")
# # # # # # # # # # # # # # # # stopword = set(stopwords.words('english'))

# # # # # # # # # # # # # # # # def clean(text):
# # # # # # # # # # # # # # # #     text = str(text).lower()
# # # # # # # # # # # # # # # #     text = re.sub(r'\[.*?\]', '', text)
# # # # # # # # # # # # # # # #     text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
# # # # # # # # # # # # # # # #     text = re.sub(r'\w*\d\w*', '', text)
# # # # # # # # # # # # # # # #     text = re.sub(r'[''""…]', '', text)
# # # # # # # # # # # # # # # #     text = re.sub(r'\n', '', text)
# # # # # # # # # # # # # # # #     text = [word for word in text.split(' ') if word not in stopword]
# # # # # # # # # # # # # # # #     text = ' '.join(text)
# # # # # # # # # # # # # # # #     text = [stemmer.stem(word) for word in text.split(' ')]
# # # # # # # # # # # # # # # #     text = ' '.join(text)
# # # # # # # # # # # # # # # #     return text

# # # # # # # # # # # # # # # # df['text'] = df['text'].apply(clean)

# # # # # # # # # # # # # # # # # Feature extraction
# # # # # # # # # # # # # # # # tfidf = TfidfVectorizer(max_features=5000)
# # # # # # # # # # # # # # # # X = tfidf.fit_transform(df['text']).toarray()
# # # # # # # # # # # # # # # # y = df['label']

# # # # # # # # # # # # # # # # # Train model
# # # # # # # # # # # # # # # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # # # # # # # # # # # # # # # model = MultinomialNB()
# # # # # # # # # # # # # # # # model.fit(X_train, y_train)

# # # # # # # # # # # # # # # # # Custom CSS for styling
# # # # # # # # # # # # # # # # st.markdown("""
# # # # # # # # # # # # # # # #     <style>
# # # # # # # # # # # # # # # #     .chat-container {
# # # # # # # # # # # # # # # #         display: flex;
# # # # # # # # # # # # # # # #         flex-direction: row;
# # # # # # # # # # # # # # # #         padding: 2rem;
# # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # #     .sidebar {
# # # # # # # # # # # # # # # #         flex: 1;
# # # # # # # # # # # # # # # #         padding: 0 2rem;
# # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # #     .main-content {
# # # # # # # # # # # # # # # #         flex: 3;
# # # # # # # # # # # # # # # #         padding: 0 2rem;
# # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # #     .message-container {
# # # # # # # # # # # # # # # #         margin-bottom: 1rem;
# # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # #     .message {
# # # # # # # # # # # # # # # #         background-color: #ffffff;
# # # # # # # # # # # # # # # #         padding: 1rem;
# # # # # # # # # # # # # # # #         border-radius: 10px;
# # # # # # # # # # # # # # # #         margin-bottom: 0.5rem;
# # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # #     .bot {
# # # # # # # # # # # # # # # #         text-align: left;
# # # # # # # # # # # # # # # #         color: #2c3e50;
# # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # #     .user {
# # # # # # # # # # # # # # # #         text-align: right;
# # # # # # # # # # # # # # # #         color: #2980b9;
# # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # #     .button {
# # # # # # # # # # # # # # # #         background-color: #3498db;
# # # # # # # # # # # # # # # #         color: white;
# # # # # # # # # # # # # # # #         border: none;
# # # # # # # # # # # # # # # #         padding: 1rem 2rem;
# # # # # # # # # # # # # # # #         font-size: 1.2rem;
# # # # # # # # # # # # # # # #         border-radius: 5px;
# # # # # # # # # # # # # # # #         cursor: pointer;
# # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # #     .button:hover {
# # # # # # # # # # # # # # # #         background-color: #2980b9;
# # # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # # #     </style>
# # # # # # # # # # # # # # # # """, unsafe_allow_html=True)

# # # # # # # # # # # # # # # # # Initialize session state for chat history
# # # # # # # # # # # # # # # # class SessionState:
# # # # # # # # # # # # # # # #     chat_history = []

# # # # # # # # # # # # # # # # session_state = SessionState()

# # # # # # # # # # # # # # # # # Streamlit app layout
# # # # # # # # # # # # # # # # st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# # # # # # # # # # # # # # # # # Sidebar for chat history
# # # # # # # # # # # # # # # # st.sidebar.title("Chat History")
# # # # # # # # # # # # # # # # chat_history = st.sidebar.empty()

# # # # # # # # # # # # # # # # st.write("""
# # # # # # # # # # # # # # # # ### Stress Detection Chatbot
# # # # # # # # # # # # # # # # Hi there! I'm here to help you detect stress. Go ahead and type your message below.
# # # # # # # # # # # # # # # # """)

# # # # # # # # # # # # # # # # # Chat interaction
# # # # # # # # # # # # # # # # user_input = st.text_input("You:", "")
# # # # # # # # # # # # # # # # if st.button("Send"):
# # # # # # # # # # # # # # # #     if user_input:
# # # # # # # # # # # # # # # #         st.markdown('<div class="message-container">', unsafe_allow_html=True)
# # # # # # # # # # # # # # # #         st.markdown('<div class="message user">' + user_input + '</div>', unsafe_allow_html=True)
# # # # # # # # # # # # # # # #         cleaned_input = clean(user_input)
# # # # # # # # # # # # # # # #         vectorized_input = tfidf.transform([cleaned_input]).toarray()
# # # # # # # # # # # # # # # #         prediction = model.predict(vectorized_input)
# # # # # # # # # # # # # # # #         result = "Stress Detected" if prediction[0] == 1 else "No Stress Detected"
# # # # # # # # # # # # # # # #         st.markdown('<div class="message bot">' + result + '</div>', unsafe_allow_html=True)
# # # # # # # # # # # # # # # #         st.markdown('</div>', unsafe_allow_html=True)

# # # # # # # # # # # # # # # #         # Update chat history
# # # # # # # # # # # # # # # #         session_state.chat_history.append({'user': user_input, 'bot': result})
# # # # # # # # # # # # # # # #         chat_history.markdown('')
# # # # # # # # # # # # # # # #         for message in session_state.chat_history:
# # # # # # # # # # # # # # # #             chat_history.markdown('<div class="message-container">', unsafe_allow_html=True)
# # # # # # # # # # # # # # # #             chat_history.markdown('<div class="message user">' + message['user'] + '</div>', unsafe_allow_html=True)
# # # # # # # # # # # # # # # #             chat_history.markdown('<div class="message bot">' + message['bot'] + '</div>', unsafe_allow_html=True)
# # # # # # # # # # # # # # # #             chat_history.markdown('</div>', unsafe_allow_html=True)

# # # # # # # # # # # # # # # # # Display word cloud
# # # # # # # # # # # # # # # # st.write("### Word Cloud of the Text Data")
# # # # # # # # # # # # # # # # wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['text']))
# # # # # # # # # # # # # # # # plt.figure(figsize=(10, 5))
# # # # # # # # # # # # # # # # plt.imshow(wordcloud, interpolation='bilinear')
# # # # # # # # # # # # # # # # plt.axis('off')
# # # # # # # # # # # # # # # # st.pyplot(plt)

# # # # # # # # # # # # # # # # # Add some information about the app
# # # # # # # # # # # # # # # # st.write("""
# # # # # # # # # # # # # # # # #### How it works:
# # # # # # # # # # # # # # # # - This chatbot uses a Naive Bayes classifier trained on textual data to detect stress.
# # # # # # # # # # # # # # # # - Type any message above and click on 'Send' to see if stress is detected.
# # # # # # # # # # # # # # # # - The model preprocesses the text by cleaning it and converting it into numerical features using TF-IDF vectorization.
# # # # # # # # # # # # # # # # """)

# # # # # # # # # # # # # # # # # Footer
# # # # # # # # # # # # # # # # st.markdown("""
# # # # # # # # # # # # # # # #     <div style='text-align: center; padding: 1rem; font-size: 0.9rem; color: #7f8c8d;'>
# # # # # # # # # # # # # # # #     Developed by [Your Name] - Final Year Project
# # # # # # # # # # # # # # # #     </div>
# # # # # # # # # # # # # # # # """, unsafe_allow_html=True)

# # # # # # # # # # # # # # # # st.markdown('</div>', unsafe_allow_html=True)



# # # # # # # # # # # # # # # import streamlit as st
# # # # # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # # # # import pandas as pd
# # # # # # # # # # # # # # # from sklearn.feature_extraction.text import TfidfVectorizer
# # # # # # # # # # # # # # # from sklearn.naive_bayes import MultinomialNB
# # # # # # # # # # # # # # # from sklearn.model_selection import train_test_split
# # # # # # # # # # # # # # # import nltk
# # # # # # # # # # # # # # # import re
# # # # # # # # # # # # # # # from nltk.corpus import stopwords
# # # # # # # # # # # # # # # import string
# # # # # # # # # # # # # # # from wordcloud import WordCloud
# # # # # # # # # # # # # # # import matplotlib.pyplot as plt

# # # # # # # # # # # # # # # # Download stopwords
# # # # # # # # # # # # # # # nltk.download('stopwords')

# # # # # # # # # # # # # # # # Load data
# # # # # # # # # # # # # # # df = pd.read_csv('stress.csv')

# # # # # # # # # # # # # # # # Preprocess data
# # # # # # # # # # # # # # # stemmer = nltk.SnowballStemmer("english")
# # # # # # # # # # # # # # # stopword = set(stopwords.words('english'))

# # # # # # # # # # # # # # # def clean(text):
# # # # # # # # # # # # # # #     text = str(text).lower()
# # # # # # # # # # # # # # #     text = re.sub(r'\[.*?\]', '', text)
# # # # # # # # # # # # # # #     text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
# # # # # # # # # # # # # # #     text = re.sub(r'\w*\d\w*', '', text)
# # # # # # # # # # # # # # #     text = re.sub(r'[''""…]', '', text)
# # # # # # # # # # # # # # #     text = re.sub(r'\n', '', text)
# # # # # # # # # # # # # # #     text = [word for word in text.split(' ') if word not in stopword]
# # # # # # # # # # # # # # #     text = ' '.join(text)
# # # # # # # # # # # # # # #     text = [stemmer.stem(word) for word in text.split(' ')]
# # # # # # # # # # # # # # #     text = ' '.join(text)
# # # # # # # # # # # # # # #     return text

# # # # # # # # # # # # # # # df['text'] = df['text'].apply(clean)

# # # # # # # # # # # # # # # # Feature extraction
# # # # # # # # # # # # # # # tfidf = TfidfVectorizer(max_features=5000)
# # # # # # # # # # # # # # # X = tfidf.fit_transform(df['text']).toarray()
# # # # # # # # # # # # # # # y = df['label']

# # # # # # # # # # # # # # # # Train model
# # # # # # # # # # # # # # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # # # # # # # # # # # # # # model = MultinomialNB()
# # # # # # # # # # # # # # # model.fit(X_train, y_train)

# # # # # # # # # # # # # # # # Custom CSS for styling
# # # # # # # # # # # # # # # st.markdown("""
# # # # # # # # # # # # # # #     <style>
# # # # # # # # # # # # # # #     .chat-container {
# # # # # # # # # # # # # # #         display: flex;
# # # # # # # # # # # # # # #         flex-direction: row;
# # # # # # # # # # # # # # #         padding: 2rem;
# # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # #     .sidebar {
# # # # # # # # # # # # # # #         flex: 1;
# # # # # # # # # # # # # # #         padding: 0 2rem;
# # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # #     .main-content {
# # # # # # # # # # # # # # #         flex: 3;
# # # # # # # # # # # # # # #         padding: 0 2rem;
# # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # #     .message-container {
# # # # # # # # # # # # # # #         margin-bottom: 1rem;
# # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # #     .message {
# # # # # # # # # # # # # # #         background-color: #ffffff;
# # # # # # # # # # # # # # #         padding: 1rem;
# # # # # # # # # # # # # # #         border-radius: 10px;
# # # # # # # # # # # # # # #         margin-bottom: 0.5rem;
# # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # #     .bot {
# # # # # # # # # # # # # # #         text-align: left;
# # # # # # # # # # # # # # #         color: #2c3e50;
# # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # #     .user {
# # # # # # # # # # # # # # #         text-align: right;
# # # # # # # # # # # # # # #         color: #2980b9;
# # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # #     .button {
# # # # # # # # # # # # # # #         background-color: #3498db;
# # # # # # # # # # # # # # #         color: white;
# # # # # # # # # # # # # # #         border: none;
# # # # # # # # # # # # # # #         padding: 1rem 2rem;
# # # # # # # # # # # # # # #         font-size: 1.2rem;
# # # # # # # # # # # # # # #         border-radius: 5px;
# # # # # # # # # # # # # # #         cursor: pointer;
# # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # #     .button:hover {
# # # # # # # # # # # # # # #         background-color: #2980b9;
# # # # # # # # # # # # # # #     }
# # # # # # # # # # # # # # #     </style>
# # # # # # # # # # # # # # # """, unsafe_allow_html=True)

# # # # # # # # # # # # # # # # Initialize session state for chat history
# # # # # # # # # # # # # # # class SessionState:
# # # # # # # # # # # # # # #     chat_history = []

# # # # # # # # # # # # # # # session_state = SessionState()

# # # # # # # # # # # # # # # # Streamlit app layout
# # # # # # # # # # # # # # # st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# # # # # # # # # # # # # # # # Sidebar for chat history
# # # # # # # # # # # # # # # st.sidebar.title("Chat History")
# # # # # # # # # # # # # # # chat_history = st.sidebar.empty()

# # # # # # # # # # # # # # # st.write("""
# # # # # # # # # # # # # # # ### Stress Detection Chatbot
# # # # # # # # # # # # # # # Hi there! I'm here to help you detect stress. Go ahead and type your message below.
# # # # # # # # # # # # # # # """)

# # # # # # # # # # # # # # # # Chat interaction
# # # # # # # # # # # # # # # user_input = st.text_input("You:", "")
# # # # # # # # # # # # # # # if st.button("Send"):
# # # # # # # # # # # # # # #     if user_input:
# # # # # # # # # # # # # # #         st.markdown('<div class="message-container">', unsafe_allow_html=True)
# # # # # # # # # # # # # # #         st.markdown('<div class="message user">' + user_input + '</div>', unsafe_allow_html=True)
# # # # # # # # # # # # # # #         cleaned_input = clean(user_input)
# # # # # # # # # # # # # # #         vectorized_input = tfidf.transform([cleaned_input]).toarray()
# # # # # # # # # # # # # # #         prediction = model.predict(vectorized_input)
# # # # # # # # # # # # # # #         result = "Stress Detected" if prediction[0] == 1 else "No Stress Detected"
# # # # # # # # # # # # # # #         st.markdown('<div class="message bot">' + result + '</div>', unsafe_allow_html=True)
# # # # # # # # # # # # # # #         st.markdown('</div>', unsafe_allow_html=True)

# # # # # # # # # # # # # # #         # Update chat history
# # # # # # # # # # # # # # #         session_state.chat_history.append({'user': user_input, 'bot': result})
# # # # # # # # # # # # # # #         chat_history.markdown('')
# # # # # # # # # # # # # # #         for message in session_state.chat_history:
# # # # # # # # # # # # # # #             chat_history.markdown('<div class="message-container">', unsafe_allow_html=True)
# # # # # # # # # # # # # # #             chat_history.markdown('<div class="message user">' + message['user'] + '</div>', unsafe_allow_html=True)
# # # # # # # # # # # # # # #             chat_history.markdown('<div class="message bot">' + message['bot'] + '</div>', unsafe_allow_html=True)
# # # # # # # # # # # # # # #             chat_history.markdown('</div>', unsafe_allow_html=True)

# # # # # # # # # # # # # # # # Display word cloud
# # # # # # # # # # # # # # # st.write("### Word Cloud of the Text Data")
# # # # # # # # # # # # # # # wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['text']))
# # # # # # # # # # # # # # # plt.figure(figsize=(10, 5))
# # # # # # # # # # # # # # # plt.imshow(wordcloud, interpolation='bilinear')
# # # # # # # # # # # # # # # plt.axis('off')
# # # # # # # # # # # # # # # st.pyplot(plt)

# # # # # # # # # # # # # # # # Add some information about the app
# # # # # # # # # # # # # # # st.write("""
# # # # # # # # # # # # # # # #### How it works:
# # # # # # # # # # # # # # # - This chatbot uses a Naive Bayes classifier trained on textual data to detect stress.
# # # # # # # # # # # # # # # - Type any message above and click on 'Send' to see if stress is detected.
# # # # # # # # # # # # # # # - The model preprocesses the text by cleaning it and converting it into numerical features using TF-IDF vectorization.
# # # # # # # # # # # # # # # """)

# # # # # # # # # # # # # # # # Footer
# # # # # # # # # # # # # # # st.markdown("""
# # # # # # # # # # # # # # #     <div style='text-align: center; padding: 1rem; font-size: 0.9rem; color: #7f8c8d;'>
# # # # # # # # # # # # # # #     Developed by [Your Name] - Final Year Project
# # # # # # # # # # # # # # #     </div>
# # # # # # # # # # # # # # # """, unsafe_allow_html=True)

# # # # # # # # # # # # # # # st.markdown('</div>', unsafe_allow_html=True)


# # # # # # # # # # # # # # import streamlit as st
# # # # # # # # # # # # # # import pandas as pd
# # # # # # # # # # # # # # from sklearn.feature_extraction.text import TfidfVectorizer
# # # # # # # # # # # # # # from sklearn.naive_bayes import MultinomialNB
# # # # # # # # # # # # # # from sklearn.model_selection import train_test_split
# # # # # # # # # # # # # # import nltk
# # # # # # # # # # # # # # import re
# # # # # # # # # # # # # # from nltk.corpus import stopwords
# # # # # # # # # # # # # # import string
# # # # # # # # # # # # # # from wordcloud import WordCloud
# # # # # # # # # # # # # # import matplotlib.pyplot as plt
# # # # # # # # # # # # # # import os

# # # # # # # # # # # # # # # Download stopwords
# # # # # # # # # # # # # # nltk.download('stopwords')

# # # # # # # # # # # # # # # Load data
# # # # # # # # # # # # # # df = pd.read_csv('stress.csv')

# # # # # # # # # # # # # # # Preprocess data
# # # # # # # # # # # # # # stemmer = nltk.SnowballStemmer("english")
# # # # # # # # # # # # # # stopword = set(stopwords.words('english'))

# # # # # # # # # # # # # # def clean(text):
# # # # # # # # # # # # # #     text = str(text).lower()
# # # # # # # # # # # # # #     text = re.sub(r'\[.*?\]', '', text)
# # # # # # # # # # # # # #     text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
# # # # # # # # # # # # # #     text = re.sub(r'\w*\d\w*', '', text)
# # # # # # # # # # # # # #     text = re.sub(r'[''""…]', '', text)
# # # # # # # # # # # # # #     text = re.sub(r'\n', '', text)
# # # # # # # # # # # # # #     text = [word for word in text.split(' ') if word not in stopword]
# # # # # # # # # # # # # #     text = ' '.join(text)
# # # # # # # # # # # # # #     text = [stemmer.stem(word) for word in text.split(' ')]
# # # # # # # # # # # # # #     text = ' '.join(text)
# # # # # # # # # # # # # #     return text

# # # # # # # # # # # # # # df['text'] = df['text'].apply(clean)

# # # # # # # # # # # # # # # Feature extraction
# # # # # # # # # # # # # # tfidf = TfidfVectorizer(max_features=5000)
# # # # # # # # # # # # # # X = tfidf.fit_transform(df['text']).toarray()
# # # # # # # # # # # # # # y = df['label']

# # # # # # # # # # # # # # # Train model
# # # # # # # # # # # # # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # # # # # # # # # # # # # model = MultinomialNB()
# # # # # # # # # # # # # # model.fit(X_train, y_train)

# # # # # # # # # # # # # # # Initialize chat history file
# # # # # # # # # # # # # # CHAT_HISTORY_FILE = 'chat_history.csv'
# # # # # # # # # # # # # # if not os.path.exists(CHAT_HISTORY_FILE):
# # # # # # # # # # # # # #     pd.DataFrame(columns=['User', 'Bot']).to_csv(CHAT_HISTORY_FILE, index=False)

# # # # # # # # # # # # # # # Custom CSS for styling
# # # # # # # # # # # # # # st.markdown("""
# # # # # # # # # # # # # #     <style>
# # # # # # # # # # # # # #     .message-container {
# # # # # # # # # # # # # #         margin-bottom: 1rem;
# # # # # # # # # # # # # #     }
# # # # # # # # # # # # # #     .message {
# # # # # # # # # # # # # #         background-color: #ffffff;
# # # # # # # # # # # # # #         padding: 1rem;
# # # # # # # # # # # # # #         border-radius: 10px;
# # # # # # # # # # # # # #         margin-bottom: 0.5rem;
# # # # # # # # # # # # # #     }
# # # # # # # # # # # # # #     .bot {
# # # # # # # # # # # # # #         text-align: left;
# # # # # # # # # # # # # #         color: #2c3e50;
# # # # # # # # # # # # # #     }
# # # # # # # # # # # # # #     .user {
# # # # # # # # # # # # # #         text-align: right;
# # # # # # # # # # # # # #         color: #2980b9;
# # # # # # # # # # # # # #     }
# # # # # # # # # # # # # #     .button {
# # # # # # # # # # # # # #         background-color: #3498db;
# # # # # # # # # # # # # #         color: white;
# # # # # # # # # # # # # #         border: none;
# # # # # # # # # # # # # #         padding: 1rem 2rem;
# # # # # # # # # # # # # #         font-size: 1.2rem;
# # # # # # # # # # # # # #         border-radius: 5px;
# # # # # # # # # # # # # #         cursor: pointer;
# # # # # # # # # # # # # #     }
# # # # # # # # # # # # # #     .button:hover {
# # # # # # # # # # # # # #         background-color: #2980b9;
# # # # # # # # # # # # # #     }
# # # # # # # # # # # # # #     </style>
# # # # # # # # # # # # # # """, unsafe_allow_html=True)

# # # # # # # # # # # # # # # Streamlit app layout
# # # # # # # # # # # # # # st.write("""
# # # # # # # # # # # # # # ### Stress Detection Chatbot
# # # # # # # # # # # # # # Hi there! I'm here to help you detect stress. Go ahead and type your message below.
# # # # # # # # # # # # # # """)

# # # # # # # # # # # # # # # Chat interaction
# # # # # # # # # # # # # # user_input = st.text_input("You:", "")
# # # # # # # # # # # # # # if st.button("Send"):
# # # # # # # # # # # # # #     if user_input:
# # # # # # # # # # # # # #         cleaned_input = clean(user_input)
# # # # # # # # # # # # # #         vectorized_input = tfidf.transform([cleaned_input]).toarray()
# # # # # # # # # # # # # #         prediction = model.predict(vectorized_input)
# # # # # # # # # # # # # #         result = "Stress Detected" if prediction[0] == 1 else "No Stress Detected"

# # # # # # # # # # # # # #         # Update chat history file
# # # # # # # # # # # # # #         chat_history_df = pd.read_csv(CHAT_HISTORY_FILE)
# # # # # # # # # # # # # #         chat_history_df = chat_history_df.append({'User': user_input, 'Bot': result}, ignore_index=True)
# # # # # # # # # # # # # #         chat_history_df.to_csv(CHAT_HISTORY_FILE, index=False)

# # # # # # # # # # # # # # # Display chat history
# # # # # # # # # # # # # # st.write("### Chat History")
# # # # # # # # # # # # # # chat_history_df = pd.read_csv(CHAT_HISTORY_FILE)
# # # # # # # # # # # # # # for index, row in chat_history_df.iterrows():
# # # # # # # # # # # # # #     st.markdown('<div class="message-container">', unsafe_allow_html=True)
# # # # # # # # # # # # # #     st.markdown('<div class="message user">' + row['User'] + '</div>', unsafe_allow_html=True)
# # # # # # # # # # # # # #     st.markdown('<div class="message bot">' + row['Bot'] + '</div>', unsafe_allow_html=True)
# # # # # # # # # # # # # #     st.markdown('</div>', unsafe_allow_html=True)

# # # # # # # # # # # # # # # Display word cloud
# # # # # # # # # # # # # # st.write("### Word Cloud of the Text Data")
# # # # # # # # # # # # # # wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['text']))
# # # # # # # # # # # # # # plt.figure(figsize=(10, 5))
# # # # # # # # # # # # # # plt.imshow(wordcloud, interpolation='bilinear')
# # # # # # # # # # # # # # plt.axis('off')
# # # # # # # # # # # # # # st.pyplot(plt)

# # # # # # # # # # # # # # # Add some information about the app
# # # # # # # # # # # # # # st.write("""
# # # # # # # # # # # # # # #### How it works:
# # # # # # # # # # # # # # - This chatbot uses a Naive Bayes classifier trained on textual data to detect stress.
# # # # # # # # # # # # # # - Type any message above and click on 'Send' to see if stress is detected.
# # # # # # # # # # # # # # - The model preprocesses the text by cleaning it and converting it into numerical features using TF-IDF vectorization.
# # # # # # # # # # # # # # """)

# # # # # # # # # # # # # # # Footer
# # # # # # # # # # # # # # st.markdown("""
# # # # # # # # # # # # # #     <div style='text-align: center; padding: 1rem; font-size: 0.9rem; color: #7f8c8d;'>
# # # # # # # # # # # # # #     Developed by [Your Name] - Final Year Project
# # # # # # # # # # # # # #     </div>
# # # # # # # # # # # # # # """, unsafe_allow_html=True)




# # # # # # # # # # # # # import streamlit as st
# # # # # # # # # # # # # import pandas as pd
# # # # # # # # # # # # # from sklearn.feature_extraction.text import TfidfVectorizer
# # # # # # # # # # # # # from sklearn.naive_bayes import MultinomialNB
# # # # # # # # # # # # # from sklearn.model_selection import train_test_split
# # # # # # # # # # # # # import nltk
# # # # # # # # # # # # # import re
# # # # # # # # # # # # # from nltk.corpus import stopwords
# # # # # # # # # # # # # import string
# # # # # # # # # # # # # from wordcloud import WordCloud
# # # # # # # # # # # # # import matplotlib.pyplot as plt
# # # # # # # # # # # # # import os

# # # # # # # # # # # # # # Download stopwords
# # # # # # # # # # # # # nltk.download('stopwords')

# # # # # # # # # # # # # # Load data
# # # # # # # # # # # # # df = pd.read_csv('stress.csv')

# # # # # # # # # # # # # # Preprocess data
# # # # # # # # # # # # # stemmer = nltk.SnowballStemmer("english")
# # # # # # # # # # # # # stopword = set(stopwords.words('english'))

# # # # # # # # # # # # # def clean(text):
# # # # # # # # # # # # #     text = str(text).lower()
# # # # # # # # # # # # #     text = re.sub(r'\[.*?\]', '', text)
# # # # # # # # # # # # #     text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
# # # # # # # # # # # # #     text = re.sub(r'\w*\d\w*', '', text)
# # # # # # # # # # # # #     text = re.sub(r'[''""…]', '', text)
# # # # # # # # # # # # #     text = re.sub(r'\n', '', text)
# # # # # # # # # # # # #     text = [word for word in text.split(' ') if word not in stopword]
# # # # # # # # # # # # #     text = ' '.join(text)
# # # # # # # # # # # # #     text = [stemmer.stem(word) for word in text.split(' ')]
# # # # # # # # # # # # #     text = ' '.join(text)
# # # # # # # # # # # # #     return text

# # # # # # # # # # # # # df['text'] = df['text'].apply(clean)

# # # # # # # # # # # # # # Feature extraction
# # # # # # # # # # # # # tfidf = TfidfVectorizer(max_features=5000)
# # # # # # # # # # # # # X = tfidf.fit_transform(df['text']).toarray()
# # # # # # # # # # # # # y = df['label']

# # # # # # # # # # # # # # Train model
# # # # # # # # # # # # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # # # # # # # # # # # # model = MultinomialNB()
# # # # # # # # # # # # # model.fit(X_train, y_train)

# # # # # # # # # # # # # # Initialize chat history file
# # # # # # # # # # # # # CHAT_HISTORY_FILE = 'chat_history.csv'
# # # # # # # # # # # # # if not os.path.exists(CHAT_HISTORY_FILE):
# # # # # # # # # # # # #     pd.DataFrame(columns=['User', 'Bot']).to_csv(CHAT_HISTORY_FILE, index=False)

# # # # # # # # # # # # # # Custom CSS for styling
# # # # # # # # # # # # # st.markdown("""
# # # # # # # # # # # # #     <style>
# # # # # # # # # # # # #     .message-container {
# # # # # # # # # # # # #         margin-bottom: 1rem;
# # # # # # # # # # # # #     }
# # # # # # # # # # # # #     .message {
# # # # # # # # # # # # #         background-color: #ffffff;
# # # # # # # # # # # # #         padding: 1rem;
# # # # # # # # # # # # #         border-radius: 10px;
# # # # # # # # # # # # #         margin-bottom: 0.5rem;
# # # # # # # # # # # # #     }
# # # # # # # # # # # # #     .bot {
# # # # # # # # # # # # #         text-align: left;
# # # # # # # # # # # # #         color: #2c3e50;
# # # # # # # # # # # # #     }
# # # # # # # # # # # # #     .user {
# # # # # # # # # # # # #         text-align: right;
# # # # # # # # # # # # #         color: #2980b9;
# # # # # # # # # # # # #     }
# # # # # # # # # # # # #     .button {
# # # # # # # # # # # # #         background-color: #3498db;
# # # # # # # # # # # # #         color: white;
# # # # # # # # # # # # #         border: none;
# # # # # # # # # # # # #         padding: 1rem 2rem;
# # # # # # # # # # # # #         font-size: 1.2rem;
# # # # # # # # # # # # #         border-radius: 5px;
# # # # # # # # # # # # #         cursor: pointer;
# # # # # # # # # # # # #     }
# # # # # # # # # # # # #     .button:hover {
# # # # # # # # # # # # #         background-color: #2980b9;
# # # # # # # # # # # # #     }
# # # # # # # # # # # # #     </style>
# # # # # # # # # # # # # """, unsafe_allow_html=True)

# # # # # # # # # # # # # # Streamlit app layout
# # # # # # # # # # # # # st.write("""
# # # # # # # # # # # # # ### Stress Detection Chatbot
# # # # # # # # # # # # # Hi there! I'm here to help you detect stress. Go ahead and type your message below.
# # # # # # # # # # # # # """)

# # # # # # # # # # # # # # Chat interaction
# # # # # # # # # # # # # user_input = st.text_input("You:", "")
# # # # # # # # # # # # # if st.button("Send"):
# # # # # # # # # # # # #     if user_input:
# # # # # # # # # # # # #         cleaned_input = clean(user_input)
# # # # # # # # # # # # #         vectorized_input = tfidf.transform([cleaned_input]).toarray()
# # # # # # # # # # # # #         prediction = model.predict(vectorized_input)
# # # # # # # # # # # # #         result = "Stress Detected" if prediction[0] == 1 else "No Stress Detected"

# # # # # # # # # # # # #         # Update chat history file
# # # # # # # # # # # # #         chat_history_df = pd.read_csv(CHAT_HISTORY_FILE)
# # # # # # # # # # # # #         chat_history_df = chat_history_df.append({'User': user_input, 'Bot': result}, ignore_index=True)
# # # # # # # # # # # # #         chat_history_df.to_csv(CHAT_HISTORY_FILE, index=False)

# # # # # # # # # # # # # # Display chat history
# # # # # # # # # # # # # st.sidebar.title("Chat History")
# # # # # # # # # # # # # chat_history_df = pd.read_csv(CHAT_HISTORY_FILE)
# # # # # # # # # # # # # for index, row in chat_history_df.iterrows():
# # # # # # # # # # # # #     st.sidebar.markdown('<div class="message-container">', unsafe_allow_html=True)
# # # # # # # # # # # # #     st.sidebar.markdown('<div class="message user">' + row['User'] + '</div>', unsafe_allow_html=True)
# # # # # # # # # # # # #     st.sidebar.markdown('<div class="message bot">' + row['Bot'] + '</div>', unsafe_allow_html=True)
# # # # # # # # # # # # #     st.sidebar.markdown('</div>', unsafe_allow_html=True)

# # # # # # # # # # # # # # Display word cloud
# # # # # # # # # # # # # st.write("### Word Cloud of the Text Data")
# # # # # # # # # # # # # wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['text']))
# # # # # # # # # # # # # plt.figure(figsize=(10, 5))
# # # # # # # # # # # # # plt.imshow(wordcloud, interpolation='bilinear')
# # # # # # # # # # # # # plt.axis('off')
# # # # # # # # # # # # # st.pyplot(plt)

# # # # # # # # # # # # # # Add some information about the app
# # # # # # # # # # # # # st.write("""
# # # # # # # # # # # # # #### How it works:
# # # # # # # # # # # # # - This chatbot uses a Naive Bayes classifier trained on textual data to detect stress.
# # # # # # # # # # # # # - Type any message above and click on 'Send' to see if stress is detected.
# # # # # # # # # # # # # - The model preprocesses the text by cleaning it and converting it into numerical features using TF-IDF vectorization.
# # # # # # # # # # # # # """)

# # # # # # # # # # # # # # Footer
# # # # # # # # # # # # # st.markdown("""
# # # # # # # # # # # # #     <div style='text-align: center; padding: 1rem; font-size: 0.9rem; color: #7f8c8d;'>
# # # # # # # # # # # # #     Developed by [Your Name] - Final Year Project
# # # # # # # # # # # # #     </div>
# # # # # # # # # # # # # """, unsafe_allow_html=True)


# # # # # # # # # # # # import streamlit as st
# # # # # # # # # # # # import pandas as pd
# # # # # # # # # # # # from sklearn.feature_extraction.text import TfidfVectorizer
# # # # # # # # # # # # from sklearn.naive_bayes import MultinomialNB
# # # # # # # # # # # # from sklearn.model_selection import train_test_split
# # # # # # # # # # # # import nltk
# # # # # # # # # # # # import re
# # # # # # # # # # # # from nltk.corpus import stopwords
# # # # # # # # # # # # import string
# # # # # # # # # # # # from wordcloud import WordCloud
# # # # # # # # # # # # import matplotlib.pyplot as plt
# # # # # # # # # # # # import os

# # # # # # # # # # # # # Download stopwords
# # # # # # # # # # # # nltk.download('stopwords')

# # # # # # # # # # # # # Load data
# # # # # # # # # # # # df = pd.read_csv('stress.csv')

# # # # # # # # # # # # # Preprocess data
# # # # # # # # # # # # stemmer = nltk.SnowballStemmer("english")
# # # # # # # # # # # # stopword = set(stopwords.words('english'))

# # # # # # # # # # # # def clean(text):
# # # # # # # # # # # #     text = str(text).lower()
# # # # # # # # # # # #     text = re.sub(r'\[.*?\]', '', text)
# # # # # # # # # # # #     text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
# # # # # # # # # # # #     text = re.sub(r'\w*\d\w*', '', text)
# # # # # # # # # # # #     text = re.sub(r'[''""…]', '', text)
# # # # # # # # # # # #     text = re.sub(r'\n', '', text)
# # # # # # # # # # # #     text = [word for word in text.split(' ') if word not in stopword]
# # # # # # # # # # # #     text = ' '.join(text)
# # # # # # # # # # # #     text = [stemmer.stem(word) for word in text.split(' ')]
# # # # # # # # # # # #     text = ' '.join(text)
# # # # # # # # # # # #     return text

# # # # # # # # # # # # df['text'] = df['text'].apply(clean)

# # # # # # # # # # # # # Feature extraction
# # # # # # # # # # # # tfidf = TfidfVectorizer(max_features=5000)
# # # # # # # # # # # # X = tfidf.fit_transform(df['text']).toarray()
# # # # # # # # # # # # y = df['label']

# # # # # # # # # # # # # Train model
# # # # # # # # # # # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # # # # # # # # # # # model = MultinomialNB()
# # # # # # # # # # # # model.fit(X_train, y_train)

# # # # # # # # # # # # # Initialize chat history file
# # # # # # # # # # # # CHAT_HISTORY_FILE = 'chat_history.csv'
# # # # # # # # # # # # if not os.path.exists(CHAT_HISTORY_FILE):
# # # # # # # # # # # #     pd.DataFrame(columns=['User', 'Bot']).to_csv(CHAT_HISTORY_FILE, index=False)

# # # # # # # # # # # # # Custom CSS for styling
# # # # # # # # # # # # st.markdown("""
# # # # # # # # # # # #     <style>
# # # # # # # # # # # #     .message-container {
# # # # # # # # # # # #         margin-bottom: 1rem;
# # # # # # # # # # # #     }
# # # # # # # # # # # #     .message {
# # # # # # # # # # # #         background-color: #ffffff;
# # # # # # # # # # # #         padding: 1rem;
# # # # # # # # # # # #         border-radius: 10px;
# # # # # # # # # # # #         margin-bottom: 0.5rem;
# # # # # # # # # # # #     }
# # # # # # # # # # # #     .bot {
# # # # # # # # # # # #         text-align: left;
# # # # # # # # # # # #         color: #2c3e50;
# # # # # # # # # # # #     }
# # # # # # # # # # # #     .user {
# # # # # # # # # # # #         text-align: right;
# # # # # # # # # # # #         color: #2980b9;
# # # # # # # # # # # #     }
# # # # # # # # # # # #     .button {
# # # # # # # # # # # #         background-color: #3498db;
# # # # # # # # # # # #         color: white;
# # # # # # # # # # # #         border: none;
# # # # # # # # # # # #         padding: 1rem 2rem;
# # # # # # # # # # # #         font-size: 1.2rem;
# # # # # # # # # # # #         border-radius: 5px;
# # # # # # # # # # # #         cursor: pointer;
# # # # # # # # # # # #     }
# # # # # # # # # # # #     .button:hover {
# # # # # # # # # # # #         background-color: #2980b9;
# # # # # # # # # # # #     }
# # # # # # # # # # # #     </style>
# # # # # # # # # # # # """, unsafe_allow_html=True)

# # # # # # # # # # # # # Streamlit app layout
# # # # # # # # # # # # st.write("""
# # # # # # # # # # # # ### Stress Detection Chatbot
# # # # # # # # # # # # Hi there! I'm here to help you detect stress. Go ahead and type your message below.
# # # # # # # # # # # # """)

# # # # # # # # # # # # # Chat interaction
# # # # # # # # # # # # user_input = st.text_input("You:", "")
# # # # # # # # # # # # if st.button("Send"):
# # # # # # # # # # # #     if user_input:
# # # # # # # # # # # #         cleaned_input = clean(user_input)
# # # # # # # # # # # #         vectorized_input = tfidf.transform([cleaned_input]).toarray()
# # # # # # # # # # # #         prediction = model.predict(vectorized_input)
# # # # # # # # # # # #         result = "Stress Detected" if prediction[0] == 1 else "No Stress Detected"

# # # # # # # # # # # #         # Update chat history file
# # # # # # # # # # # #         chat_history_df = pd.read_csv(CHAT_HISTORY_FILE)
# # # # # # # # # # # #         chat_history_df = pd.concat([chat_history_df, pd.DataFrame({'User': [user_input], 'Bot': [result]})], ignore_index=True)
# # # # # # # # # # # #         chat_history_df.to_csv(CHAT_HISTORY_FILE, index=False)

# # # # # # # # # # # # # Display chat history
# # # # # # # # # # # # st.sidebar.title("Chat History")
# # # # # # # # # # # # chat_history_df = pd.read_csv(CHAT_HISTORY_FILE)
# # # # # # # # # # # # for index, row in chat_history_df.iterrows():
# # # # # # # # # # # #     st.sidebar.markdown('<div class="message-container">', unsafe_allow_html=True)
# # # # # # # # # # # #     st.sidebar.markdown('<div class="message user">' + row['User'] + '</div>', unsafe_allow_html=True)
# # # # # # # # # # # #     st.sidebar.markdown('<div class="message bot">' + row['Bot'] + '</div>', unsafe_allow_html=True)
# # # # # # # # # # # #     st.sidebar.markdown('</div>', unsafe_allow_html=True)

# # # # # # # # # # # # # Display word cloud
# # # # # # # # # # # # st.write("### Word Cloud of the Text Data")
# # # # # # # # # # # # wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['text']))
# # # # # # # # # # # # plt.figure(figsize=(10, 5))
# # # # # # # # # # # # plt.imshow(wordcloud, interpolation='bilinear')
# # # # # # # # # # # # plt.axis('off')
# # # # # # # # # # # # st.pyplot(plt)

# # # # # # # # # # # # # Add some information about the app
# # # # # # # # # # # # st.write("""
# # # # # # # # # # # # #### How it works:
# # # # # # # # # # # # - This chatbot uses a Naive Bayes classifier trained on textual data to detect stress.
# # # # # # # # # # # # - Type any message above and click on 'Send' to see if stress is detected.
# # # # # # # # # # # # - The model preprocesses the text by cleaning it and converting it into numerical features using TF-IDF vectorization.
# # # # # # # # # # # # """)

# # # # # # # # # # # # # Footer
# # # # # # # # # # # # st.markdown("""
# # # # # # # # # # # #     <div style='text-align: center; padding: 1rem; font-size: 0.9rem; color: #7f8c8d;'>
# # # # # # # # # # # #     Developed by [Your Name] - Final Year Project
# # # # # # # # # # # #     </div>
# # # # # # # # # # # # """, unsafe_allow_html=True)
# # # # # # # # # # # # #shi haii ye upar wala
# # # # # # # # # # # import streamlit as st
# # # # # # # # # # # import pandas as pd
# # # # # # # # # # # from sklearn.feature_extraction.text import TfidfVectorizer
# # # # # # # # # # # from sklearn.naive_bayes import MultinomialNB
# # # # # # # # # # # from sklearn.model_selection import train_test_split
# # # # # # # # # # # import nltk
# # # # # # # # # # # import re
# # # # # # # # # # # from nltk.corpus import stopwords
# # # # # # # # # # # import string
# # # # # # # # # # # from wordcloud import WordCloud
# # # # # # # # # # # import matplotlib.pyplot as plt
# # # # # # # # # # # import os

# # # # # # # # # # # # Download stopwords
# # # # # # # # # # # nltk.download('stopwords')

# # # # # # # # # # # # Load data
# # # # # # # # # # # df = pd.read_csv('stress.csv')

# # # # # # # # # # # # Preprocess data
# # # # # # # # # # # stemmer = nltk.SnowballStemmer("english")
# # # # # # # # # # # stopword = set(stopwords.words('english'))

# # # # # # # # # # # def clean(text):
# # # # # # # # # # #     text = str(text).lower()
# # # # # # # # # # #     text = re.sub(r'\[.*?\]', '', text)
# # # # # # # # # # #     text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
# # # # # # # # # # #     text = re.sub(r'\w*\d\w*', '', text)
# # # # # # # # # # #     text = re.sub(r'[''""…]', '', text)
# # # # # # # # # # #     text = re.sub(r'\n', '', text)
# # # # # # # # # # #     text = [word for word in text.split(' ') if word not in stopword]
# # # # # # # # # # #     text = ' '.join(text)
# # # # # # # # # # #     text = [stemmer.stem(word) for word in text.split(' ')]
# # # # # # # # # # #     text = ' '.join(text)
# # # # # # # # # # #     return text

# # # # # # # # # # # df['text'] = df['text'].apply(clean)

# # # # # # # # # # # # Feature extraction
# # # # # # # # # # # tfidf = TfidfVectorizer(max_features=5000)
# # # # # # # # # # # X = tfidf.fit_transform(df['text']).toarray()
# # # # # # # # # # # y = df['label']

# # # # # # # # # # # # Train model
# # # # # # # # # # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # # # # # # # # # # model = MultinomialNB()
# # # # # # # # # # # model.fit(X_train, y_train)

# # # # # # # # # # # # Initialize chat history file
# # # # # # # # # # # CHAT_HISTORY_FILE = 'chat_history.csv'
# # # # # # # # # # # if not os.path.exists(CHAT_HISTORY_FILE):
# # # # # # # # # # #     pd.DataFrame(columns=['User', 'Bot']).to_csv(CHAT_HISTORY_FILE, index=False)

# # # # # # # # # # # # Custom CSS for styling
# # # # # # # # # # # st.markdown("""
# # # # # # # # # # #     <style>
# # # # # # # # # # #     .message-container {
# # # # # # # # # # #         margin-bottom: 1rem;
# # # # # # # # # # #     }
# # # # # # # # # # #     .message {
# # # # # # # # # # #         background-color: #ffffff;
# # # # # # # # # # #         padding: 1rem;
# # # # # # # # # # #         border-radius: 10px;
# # # # # # # # # # #         margin-bottom: 0.5rem;
# # # # # # # # # # #         font-family: cursive; /* Change font family to cursive */
# # # # # # # # # # #     }
# # # # # # # # # # #     .bot {
# # # # # # # # # # #         text-align: left;
# # # # # # # # # # #         color: #2c3e50;
# # # # # # # # # # #     }
# # # # # # # # # # #     .user {
# # # # # # # # # # #         text-align: right;
# # # # # # # # # # #         color: #2980b9;
# # # # # # # # # # #     }
# # # # # # # # # # #     .button {
# # # # # # # # # # #         background-color: #3498db;
# # # # # # # # # # #         color: white;
# # # # # # # # # # #         border: none;
# # # # # # # # # # #         padding: 1rem 2rem;
# # # # # # # # # # #         font-size: 1.2rem;
# # # # # # # # # # #         border-radius: 5px;
# # # # # # # # # # #         cursor: pointer;
# # # # # # # # # # #     }
# # # # # # # # # # #     .button:hover {
# # # # # # # # # # #         background-color: #2980b9;
# # # # # # # # # # #     }
# # # # # # # # # # #     </style>
# # # # # # # # # # # """, unsafe_allow_html=True)

# # # # # # # # # # # # Streamlit app layout
# # # # # # # # # # # st.write("""
# # # # # # # # # # # ### Stress Detection Chatbot
# # # # # # # # # # # Hi there! I'm here to help you detect stress. Go ahead and type your message below.
# # # # # # # # # # # """)

# # # # # # # # # # # # Chat interaction
# # # # # # # # # # # user_input = st.text_input("You:", "")
# # # # # # # # # # # if st.button("Send"):
# # # # # # # # # # #     if user_input:
# # # # # # # # # # #         cleaned_input = clean(user_input)
# # # # # # # # # # #         vectorized_input = tfidf.transform([cleaned_input]).toarray()
# # # # # # # # # # #         prediction = model.predict(vectorized_input)
# # # # # # # # # # #         result = "Stress Detected" if prediction[0] == 1 else "No Stress Detected"

# # # # # # # # # # #         # Update chat history file
# # # # # # # # # # #         chat_history_df = pd.read_csv(CHAT_HISTORY_FILE)
# # # # # # # # # # #         chat_history_df = pd.concat([chat_history_df, pd.DataFrame({'User': [user_input], 'Bot': [result]})], ignore_index=True)
# # # # # # # # # # #         chat_history_df.to_csv(CHAT_HISTORY_FILE, index=False)

# # # # # # # # # # # # Display chat history
# # # # # # # # # # # st.sidebar.title("Chat History")
# # # # # # # # # # # chat_history_df = pd.read_csv(CHAT_HISTORY_FILE)
# # # # # # # # # # # for index, row in chat_history_df.iterrows():
# # # # # # # # # # #     st.sidebar.markdown('<div class="message-container">', unsafe_allow_html=True)
# # # # # # # # # # #     st.sidebar.markdown('<div class="message user">' + row['User'] + '</div>', unsafe_allow_html=True)
# # # # # # # # # # #     st.sidebar.markdown('<div class="message bot">' + row['Bot'] + '</div>', unsafe_allow_html=True)
# # # # # # # # # # #     st.sidebar.markdown('</div>', unsafe_allow_html=True)

# # # # # # # # # # # # Display word cloud
# # # # # # # # # # # st.write("### Word Cloud of the Text Data")
# # # # # # # # # # # wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['text']))
# # # # # # # # # # # plt.figure(figsize=(10, 5))
# # # # # # # # # # # plt.imshow(wordcloud, interpolation='bilinear')
# # # # # # # # # # # plt.axis('off')
# # # # # # # # # # # st.pyplot(plt)

# # # # # # # # # # # # Add some information about the app
# # # # # # # # # # # st.write("""
# # # # # # # # # # # #### How it works:
# # # # # # # # # # # - This chatbot uses a Naive Bayes classifier trained on textual data to detect stress.
# # # # # # # # # # # - Type any message above and click on 'Send' to see if stress is detected.
# # # # # # # # # # # - The model preprocesses the text by cleaning it and converting it into numerical features using TF-IDF vectorization.
# # # # # # # # # # # """)

# # # # # # # # # # # # Footer
# # # # # # # # # # # st.markdown("""
# # # # # # # # # # #     <div style='text-align: center; padding: 1rem; font-size: 0.9rem; color: #7f8c8d;'>
# # # # # # # # # # #     Developed by Hassan Arzoo - Roll No: CSE204008, Guided by Dr. Abhishek Das, Associate Professor, CSE Department, Aliah University
# # # # # # # # # # #     </div>
# # # # # # # # # # # """, unsafe_allow_html=True)
# # # # # # # # # # # #shi haiiii
# # # # # # # # # # #Sure! You can add various types of graphs to your Streamlit app using libraries such as Matplotlib, Seaborn, and Plotly. Here are a few ideas for graphs that might be useful in the context of a stress detection application:

# # # # # # # # # # #1. **Distribution of Stress and Non-Stress Labels**: A bar chart or pie chart showing the distribution of stress and non-stress labels in your dataset.
# # # # # # # # # # #2. **Word Frequencies**: A bar chart showing the most common words in stress-related and non-stress-related texts.
# # # # # # # # # # #3. **Model Performance Metrics**: Graphs showing the performance metrics of your model, such as accuracy, precision, recall, and F1 score.
# # # # # # # # # # #4. **Time Series Analysis**: If you have timestamped data, a line chart showing the trend of stress levels over time.

# # # # # # # # # # #Here's how you can integrate a few of these into your existing Streamlit app:

# # # # # # # # # # ### Updated Code with Graphs


# # # # # # # # # # import streamlit as st
# # # # # # # # # # import pandas as pd
# # # # # # # # # # import numpy as np
# # # # # # # # # # from sklearn.feature_extraction.text import TfidfVectorizer
# # # # # # # # # # from sklearn.naive_bayes import MultinomialNB
# # # # # # # # # # from sklearn.model_selection import train_test_split
# # # # # # # # # # from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# # # # # # # # # # import nltk
# # # # # # # # # # import re
# # # # # # # # # # from nltk.corpus import stopwords
# # # # # # # # # # import string
# # # # # # # # # # from wordcloud import WordCloud
# # # # # # # # # # import matplotlib.pyplot as plt
# # # # # # # # # # import seaborn as sns
# # # # # # # # # # import os

# # # # # # # # # # # Download stopwords
# # # # # # # # # # nltk.download('stopwords')

# # # # # # # # # # # Load data
# # # # # # # # # # df = pd.read_csv('stress.csv')

# # # # # # # # # # # Preprocess data
# # # # # # # # # # stemmer = nltk.SnowballStemmer("english")
# # # # # # # # # # stopword = set(stopwords.words('english'))

# # # # # # # # # # def clean(text):
# # # # # # # # # #     text = str(text).lower()
# # # # # # # # # #     text = re.sub(r'\[.*?\]', '', text)
# # # # # # # # # #     text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
# # # # # # # # # #     text = re.sub(r'\w*\d\w*', '', text)
# # # # # # # # # #     text = re.sub(r'[''""…]', '', text)
# # # # # # # # # #     text = re.sub(r'\n', '', text)
# # # # # # # # # #     text = [word for word in text.split(' ') if word not in stopword]
# # # # # # # # # #     text = ' '.join(text)
# # # # # # # # # #     text = [stemmer.stem(word) for word in text.split(' ')]
# # # # # # # # # #     text = ' '.join(text)
# # # # # # # # # #     return text

# # # # # # # # # # df['text'] = df['text'].apply(clean)

# # # # # # # # # # # Feature extraction
# # # # # # # # # # tfidf = TfidfVectorizer(max_features=5000)
# # # # # # # # # # X = tfidf.fit_transform(df['text']).toarray()
# # # # # # # # # # y = df['label']

# # # # # # # # # # # Train model
# # # # # # # # # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # # # # # # # # # model = MultinomialNB()
# # # # # # # # # # model.fit(X_train, y_train)
# # # # # # # # # # y_pred = model.predict(X_test)

# # # # # # # # # # # Performance metrics
# # # # # # # # # # accuracy = accuracy_score(y_test, y_pred)
# # # # # # # # # # precision = precision_score(y_test, y_pred)
# # # # # # # # # # recall = recall_score(y_test, y_pred)
# # # # # # # # # # f1 = f1_score(y_test, y_pred)

# # # # # # # # # # # Initialize chat history file
# # # # # # # # # # CHAT_HISTORY_FILE = 'chat_history.csv'
# # # # # # # # # # if not os.path.exists(CHAT_HISTORY_FILE):
# # # # # # # # # #     pd.DataFrame(columns=['User', 'Bot']).to_csv(CHAT_HISTORY_FILE, index=False)

# # # # # # # # # # # Custom CSS for styling
# # # # # # # # # # st.markdown("""
# # # # # # # # # #     <style>
# # # # # # # # # #     .message-container {
# # # # # # # # # #         margin-bottom: 1rem;
# # # # # # # # # #     }
# # # # # # # # # #     .message {
# # # # # # # # # #         background-color: #ffffff;
# # # # # # # # # #         padding: 1rem;
# # # # # # # # # #         border-radius: 10px;
# # # # # # # # # #         margin-bottom: 0.5rem;
# # # # # # # # # #         font-family: cursive; /* Change font family to cursive */
# # # # # # # # # #     }
# # # # # # # # # #     .bot {
# # # # # # # # # #         text-align: left;
# # # # # # # # # #         color: #2c3e50;
# # # # # # # # # #     }
# # # # # # # # # #     .user {
# # # # # # # # # #         text-align: right;
# # # # # # # # # #         color: #2980b9;
# # # # # # # # # #     }
# # # # # # # # # #     .button {
# # # # # # # # # #         background-color: #3498db;
# # # # # # # # # #         color: white;
# # # # # # # # # #         border: none;
# # # # # # # # # #         padding: 1rem 2rem;
# # # # # # # # # #         font-size: 1.2rem;
# # # # # # # # # #         border-radius: 5px;
# # # # # # # # # #         cursor: pointer;
# # # # # # # # # #     }
# # # # # # # # # #     .button:hover {
# # # # # # # # # #         background-color: #2980b9;
# # # # # # # # # #     }
# # # # # # # # # #     </style>
# # # # # # # # # # """, unsafe_allow_html=True)

# # # # # # # # # # # Streamlit app layout
# # # # # # # # # # st.write("""
# # # # # # # # # # ### Stress Detection Chatbot
# # # # # # # # # # Hi there! I'm here to help you detect stress. Go ahead and type your message below.
# # # # # # # # # # """)

# # # # # # # # # # # Chat interaction
# # # # # # # # # # user_input = st.text_input("You:", "")
# # # # # # # # # # if st.button("Send"):
# # # # # # # # # #     if user_input:
# # # # # # # # # #         cleaned_input = clean(user_input)
# # # # # # # # # #         vectorized_input = tfidf.transform([cleaned_input]).toarray()
# # # # # # # # # #         prediction = model.predict(vectorized_input)
# # # # # # # # # #         result = "Stress Detected" if prediction[0] == 1 else "No Stress Detected"

# # # # # # # # # #         # Update chat history file
# # # # # # # # # #         chat_history_df = pd.read_csv(CHAT_HISTORY_FILE)
# # # # # # # # # #         chat_history_df = pd.concat([chat_history_df, pd.DataFrame({'User': [user_input], 'Bot': [result]})], ignore_index=True)
# # # # # # # # # #         chat_history_df.to_csv(CHAT_HISTORY_FILE, index=False)

# # # # # # # # # # # Display chat history
# # # # # # # # # # st.sidebar.title("Chat History")
# # # # # # # # # # chat_history_df = pd.read_csv(CHAT_HISTORY_FILE)
# # # # # # # # # # for index, row in chat_history_df.iterrows():
# # # # # # # # # #     st.sidebar.markdown('<div class="message-container">', unsafe_allow_html=True)
# # # # # # # # # #     st.sidebar.markdown('<div class="message user">' + row['User'] + '</div>', unsafe_allow_html=True)
# # # # # # # # # #     st.sidebar.markdown('<div class="message bot">' + row['Bot'] + '</div>', unsafe_allow_html=True)
# # # # # # # # # #     st.sidebar.markdown('</div>', unsafe_allow_html=True)

# # # # # # # # # # # Display word cloud
# # # # # # # # # # st.write("### Word Cloud of the Text Data")
# # # # # # # # # # wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['text']))
# # # # # # # # # # plt.figure(figsize=(10, 5))
# # # # # # # # # # plt.imshow(wordcloud, interpolation='bilinear')
# # # # # # # # # # plt.axis('off')
# # # # # # # # # # st.pyplot(plt)

# # # # # # # # # # # Add some information about the app
# # # # # # # # # # st.write("""
# # # # # # # # # # #### How it works:
# # # # # # # # # # - This chatbot uses a Naive Bayes classifier trained on textual data to detect stress.
# # # # # # # # # # - Type any message above and click on 'Send' to see if stress is detected.
# # # # # # # # # # - The model preprocesses the text by cleaning it and converting it into numerical features using TF-IDF vectorization.
# # # # # # # # # # """)

# # # # # # # # # # # Display performance metrics
# # # # # # # # # # st.write("### Model Performance Metrics")
# # # # # # # # # # st.write(f"Accuracy: {accuracy:.2f}")
# # # # # # # # # # st.write(f"Precision: {precision:.2f}")
# # # # # # # # # # st.write(f"Recall: {recall:.2f}")
# # # # # # # # # # st.write(f"F1 Score: {f1:.2f}")

# # # # # # # # # # # Distribution of Stress and Non-Stress Labels
# # # # # # # # # # st.write("### Distribution of Stress and Non-Stress Labels")
# # # # # # # # # # label_counts = df['label'].value_counts()
# # # # # # # # # # plt.figure(figsize=(6, 4))
# # # # # # # # # # sns.barplot(x=label_counts.index, y=label_counts.values, palette='viridis')
# # # # # # # # # # plt.xlabel('Label')
# # # # # # # # # # plt.ylabel('Count')
# # # # # # # # # # plt.title('Distribution of Stress and Non-Stress Labels')
# # # # # # # # # # st.pyplot(plt)

# # # # # # # # # # # Most common words in stress-related and non-stress-related texts
# # # # # # # # # # st.write("### Most Common Words in Stress-Related Texts")
# # # # # # # # # # stress_texts = ' '.join(df[df['label'] == 1]['text'])
# # # # # # # # # # stress_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(stress_texts)
# # # # # # # # # # plt.figure(figsize=(10, 5))
# # # # # # # # # # plt.imshow(stress_wordcloud, interpolation='bilinear')
# # # # # # # # # # plt.axis('off')
# # # # # # # # # # st.pyplot(plt)

# # # # # # # # # # st.write("### Most Common Words in Non-Stress-Related Texts")
# # # # # # # # # # non_stress_texts = ' '.join(df[df['label'] == 0]['text'])
# # # # # # # # # # non_stress_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(non_stress_texts)
# # # # # # # # # # plt.figure(figsize=(10, 5))
# # # # # # # # # # plt.imshow(non_stress_wordcloud, interpolation='bilinear')
# # # # # # # # # # plt.axis('off')
# # # # # # # # # # st.pyplot(plt)

# # # # # # # # # # # Footer
# # # # # # # # # # st.markdown("""
# # # # # # # # # #     <div style='text-align: center; padding: 1rem; font-size: 0.9rem; color: #7f8c8d;'>
# # # # # # # # # #     Developed by Hassan Arzoo - Roll No: CSE204008, Guided by Dr. Abhishek Das, Associate Professor, CSE Department, Aliah University
# # # # # # # # # #     </div>
# # # # # # # # # # """, unsafe_allow_html=True)


# # # # # # # # # import streamlit as st
# # # # # # # # # import pandas as pd
# # # # # # # # # import numpy as np
# # # # # # # # # from sklearn.feature_extraction.text import TfidfVectorizer
# # # # # # # # # from sklearn.naive_bayes import MultinomialNB
# # # # # # # # # from sklearn.model_selection import train_test_split
# # # # # # # # # from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# # # # # # # # # import nltk
# # # # # # # # # import re
# # # # # # # # # from nltk.corpus import stopwords
# # # # # # # # # import string
# # # # # # # # # from wordcloud import WordCloud
# # # # # # # # # import matplotlib.pyplot as plt
# # # # # # # # # import seaborn as sns
# # # # # # # # # import os

# # # # # # # # # # Download stopwords
# # # # # # # # # nltk.download('stopwords')

# # # # # # # # # # Load data
# # # # # # # # # df = pd.read_csv('stress.csv')

# # # # # # # # # # Preprocess data
# # # # # # # # # stemmer = nltk.SnowballStemmer("english")
# # # # # # # # # stopword = set(stopwords.words('english'))

# # # # # # # # # def clean(text):
# # # # # # # # #     text = str(text).lower()
# # # # # # # # #     text = re.sub(r'\[.*?\]', '', text)
# # # # # # # # #     text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
# # # # # # # # #     text = re.sub(r'\w*\d\w*', '', text)
# # # # # # # # #     text = re.sub(r'[''""…]', '', text)
# # # # # # # # #     text = re.sub(r'\n', '', text)
# # # # # # # # #     text = [word for word in text.split(' ') if word not in stopword]
# # # # # # # # #     text = ' '.join(text)
# # # # # # # # #     text = [stemmer.stem(word) for word in text.split(' ')]
# # # # # # # # #     text = ' '.join(text)
# # # # # # # # #     return text

# # # # # # # # # df['text'] = df['text'].apply(clean)

# # # # # # # # # # Feature extraction
# # # # # # # # # tfidf = TfidfVectorizer(max_features=5000)
# # # # # # # # # X = tfidf.fit_transform(df['text']).toarray()
# # # # # # # # # y = df['label']

# # # # # # # # # # Train model
# # # # # # # # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # # # # # # # # model = MultinomialNB()
# # # # # # # # # model.fit(X_train, y_train)
# # # # # # # # # y_pred = model.predict(X_test)

# # # # # # # # # # Performance metrics
# # # # # # # # # accuracy = accuracy_score(y_test, y_pred)
# # # # # # # # # precision = precision_score(y_test, y_pred)
# # # # # # # # # recall = recall_score(y_test, y_pred)
# # # # # # # # # f1 = f1_score(y_test, y_pred)

# # # # # # # # # # Initialize chat history file
# # # # # # # # # CHAT_HISTORY_FILE = 'chat_history.csv'
# # # # # # # # # if not os.path.exists(CHAT_HISTORY_FILE):
# # # # # # # # #     pd.DataFrame(columns=['User', 'Bot']).to_csv(CHAT_HISTORY_FILE, index=False)

# # # # # # # # # # Custom CSS for styling
# # # # # # # # # st.markdown("""
# # # # # # # # #     <style>
# # # # # # # # #     .message-container {
# # # # # # # # #         margin-bottom: 1rem;
# # # # # # # # #     }
# # # # # # # # #     .message {
# # # # # # # # #         background-color: #ffffff;
# # # # # # # # #         padding: 1rem;
# # # # # # # # #         border-radius: 10px;
# # # # # # # # #         margin-bottom: 0.5rem;
# # # # # # # # #         font-family: cursive; /* Change font family to cursive */
# # # # # # # # #     }
# # # # # # # # #     .bot {
# # # # # # # # #         text-align: left;
# # # # # # # # #         color: #2c3e50;
# # # # # # # # #     }
# # # # # # # # #     .user {
# # # # # # # # #         text-align: right;
# # # # # # # # #         color: #2980b9;
# # # # # # # # #     }
# # # # # # # # #     .button {
# # # # # # # # #         background-color: #3498db;
# # # # # # # # #         color: white;
# # # # # # # # #         border: none;
# # # # # # # # #         padding: 1rem 2rem;
# # # # # # # # #         font-size: 1.2rem;
# # # # # # # # #         border-radius: 5px;
# # # # # # # # #         cursor: pointer;
# # # # # # # # #     }
# # # # # # # # #     .button:hover {
# # # # # # # # #         background-color: #2980b9;
# # # # # # # # #     }
# # # # # # # # #     </style>
# # # # # # # # # """, unsafe_allow_html=True)

# # # # # # # # # # Streamlit app layout
# # # # # # # # # st.write("""
# # # # # # # # # ### Stress Detection Chatbot
# # # # # # # # # Hi there! I'm here to help you detect stress. Go ahead and type your message below.
# # # # # # # # # """)

# # # # # # # # # # Chat interaction
# # # # # # # # # user_input = st.text_input("You:", "")
# # # # # # # # # if st.button("Send"):
# # # # # # # # #     if user_input:
# # # # # # # # #         cleaned_input = clean(user_input)
# # # # # # # # #         vectorized_input = tfidf.transform([cleaned_input]).toarray()
# # # # # # # # #         prediction = model.predict(vectorized_input)
# # # # # # # # #         result = "Stress Detected" if prediction[0] == 1 else "No Stress Detected"

# # # # # # # # #         # Update chat history file
# # # # # # # # #         chat_history_df = pd.read_csv(CHAT_HISTORY_FILE)
# # # # # # # # #         chat_history_df = pd.concat([chat_history_df, pd.DataFrame({'User': [user_input], 'Bot': [result]})], ignore_index=True)
# # # # # # # # #         chat_history_df.to_csv(CHAT_HISTORY_FILE, index=False)

# # # # # # # # # # Display chat history
# # # # # # # # # st.sidebar.title("Chat History")
# # # # # # # # # chat_history_df = pd.read_csv(CHAT_HISTORY_FILE)
# # # # # # # # # for index, row in chat_history_df.iterrows():
# # # # # # # # #     st.sidebar.markdown('<div class="message-container">', unsafe_allow_html=True)
# # # # # # # # #     st.sidebar.markdown('<div class="message user">' + row['User'] + '</div>', unsafe_allow_html=True)
# # # # # # # # #     st.sidebar.markdown('<div class="message bot">' + row['Bot'] + '</div>', unsafe_allow_html=True)
# # # # # # # # #     st.sidebar.markdown('</div>', unsafe_allow_html=True)

# # # # # # # # # # Display chat history graph
# # # # # # # # # st.sidebar.title("Chat History Graph")
# # # # # # # # # if not chat_history_df.empty:
# # # # # # # # #     chat_history_df['Bot'] = chat_history_df['Bot'].apply(lambda x: 1 if x == "Stress Detected" else 0)
# # # # # # # # #     plt.figure(figsize=(6, 4))
# # # # # # # # #     sns.lineplot(data=chat_history_df, x=chat_history_df.index, y='Bot', markers=True)
# # # # # # # # #     plt.xlabel('Interaction')
# # # # # # # # #     plt.ylabel('Stress (1) / No Stress (0)')
# # # # # # # # #     plt.title('Chat History: Stress Detection Over Time')
# # # # # # # # #     st.sidebar.pyplot(plt)

# # # # # # # # # # Display word cloud
# # # # # # # # # st.write("### Word Cloud of the Text Data")
# # # # # # # # # wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['text']))
# # # # # # # # # plt.figure(figsize=(10, 5))
# # # # # # # # # plt.imshow(wordcloud, interpolation='bilinear')
# # # # # # # # # plt.axis('off')
# # # # # # # # # st.pyplot(plt)

# # # # # # # # # # Add some information about the app
# # # # # # # # # st.write("""
# # # # # # # # # #### How it works:
# # # # # # # # # - This chatbot uses a Naive Bayes classifier trained on textual data to detect stress.
# # # # # # # # # - Type any message above and click on 'Send' to see if stress is detected.
# # # # # # # # # - The model preprocesses the text by cleaning it and converting it into numerical features using TF-IDF vectorization.
# # # # # # # # # """)

# # # # # # # # # # Display performance metrics
# # # # # # # # # st.write("### Model Performance Metrics")
# # # # # # # # # st.write(f"Accuracy: {accuracy:.2f}")
# # # # # # # # # st.write(f"Precision: {precision:.2f}")
# # # # # # # # # st.write(f"Recall: {recall:.2f}")
# # # # # # # # # st.write(f"F1 Score: {f1:.2f}")

# # # # # # # # # # Distribution of Stress and Non-Stress Labels
# # # # # # # # # st.write("### Distribution of Stress and Non-Stress Labels")
# # # # # # # # # label_counts = df['label'].value_counts()
# # # # # # # # # plt.figure(figsize=(6, 4))
# # # # # # # # # sns.barplot(x=label_counts.index, y=label_counts.values, palette='viridis')
# # # # # # # # # plt.xlabel('Label')
# # # # # # # # # plt.ylabel('Count')
# # # # # # # # # plt.title('Distribution of Stress and Non-Stress Labels')
# # # # # # # # # st.pyplot(plt)

# # # # # # # # # # Most common words in stress-related and non-stress-related texts
# # # # # # # # # st.write("### Most Common Words in Stress-Related Texts")
# # # # # # # # # stress_texts = ' '.join(df[df['label'] == 1]['text'])
# # # # # # # # # stress_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(stress_texts)
# # # # # # # # # plt.figure(figsize=(10, 5))
# # # # # # # # # plt.imshow(stress_wordcloud, interpolation='bilinear')
# # # # # # # # # plt.axis('off')
# # # # # # # # # st.pyplot(plt)

# # # # # # # # # st.write("### Most Common Words in Non-Stress-Related Texts")
# # # # # # # # # non_stress_texts = ' '.join(df[df['label'] == 0]['text'])
# # # # # # # # # non_stress_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(non_stress_texts)
# # # # # # # # # plt.figure(figsize=(10, 5))
# # # # # # # # # plt.imshow(non_stress_wordcloud, interpolation='bilinear')
# # # # # # # # # plt.axis('off')
# # # # # # # # # st.pyplot(plt)

# # # # # # # # # # Footer
# # # # # # # # # st.markdown("""
# # # # # # # # #     <div style='text-align: center; padding: 1rem; font-size: 0.9rem; color: #7f8c8d;'>
# # # # # # # # #     Developed by Hassan Arzoo - Roll No: CSE204008, Guided by Dr. Abhishek Das, Associate Professor, CSE Department, Aliah University
# # # # # # # # #     </div>
# # # # # # # # # """, unsafe_allow_html=True)
# # # # # # # # #shi haiii 



# # # # # # # # import streamlit as st
# # # # # # # # import pandas as pd
# # # # # # # # import numpy as np
# # # # # # # # from sklearn.feature_extraction.text import TfidfVectorizer
# # # # # # # # from sklearn.naive_bayes import MultinomialNB
# # # # # # # # from sklearn.model_selection import train_test_split
# # # # # # # # from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# # # # # # # # import nltk
# # # # # # # # import re
# # # # # # # # from nltk.corpus import stopwords
# # # # # # # # import string
# # # # # # # # from wordcloud import WordCloud
# # # # # # # # import matplotlib.pyplot as plt
# # # # # # # # import seaborn as sns
# # # # # # # # import os

# # # # # # # # # Download stopwords
# # # # # # # # nltk.download('stopwords')

# # # # # # # # # Load data
# # # # # # # # df = pd.read_csv('stress.csv')

# # # # # # # # # Preprocess data
# # # # # # # # stemmer = nltk.SnowballStemmer("english")
# # # # # # # # stopword = set(stopwords.words('english'))

# # # # # # # # def clean(text):
# # # # # # # #     text = str(text).lower()
# # # # # # # #     text = re.sub(r'\[.*?\]', '', text)
# # # # # # # #     text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
# # # # # # # #     text = re.sub(r'\w*\d\w*', '', text)
# # # # # # # #     text = re.sub(r'[''""…]', '', text)
# # # # # # # #     text = re.sub(r'\n', '', text)
# # # # # # # #     text = [word for word in text.split(' ') if word not in stopword]
# # # # # # # #     text = ' '.join(text)
# # # # # # # #     text = [stemmer.stem(word) for word in text.split(' ')]
# # # # # # # #     text = ' '.join(text)
# # # # # # # #     return text

# # # # # # # # df['text'] = df['text'].apply(clean)

# # # # # # # # # Feature extraction
# # # # # # # # tfidf = TfidfVectorizer(max_features=5000)
# # # # # # # # X = tfidf.fit_transform(df['text']).toarray()
# # # # # # # # y = df['label']

# # # # # # # # # Train model
# # # # # # # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # # # # # # # model = MultinomialNB()
# # # # # # # # model.fit(X_train, y_train)
# # # # # # # # y_pred = model.predict(X_test)

# # # # # # # # # Performance metrics
# # # # # # # # accuracy = accuracy_score(y_test, y_pred)
# # # # # # # # precision = precision_score(y_test, y_pred)
# # # # # # # # recall = recall_score(y_test, y_pred)
# # # # # # # # f1 = f1_score(y_test, y_pred)

# # # # # # # # # Initialize chat history file
# # # # # # # # CHAT_HISTORY_FILE = 'chat_history.csv'
# # # # # # # # if not os.path.exists(CHAT_HISTORY_FILE):
# # # # # # # #     pd.DataFrame(columns=['User', 'Bot']).to_csv(CHAT_HISTORY_FILE, index=False)

# # # # # # # # # Custom CSS for styling
# # # # # # # # st.markdown("""
# # # # # # # #     <style>
# # # # # # # #     .message-container {
# # # # # # # #         margin-bottom: 1rem;
# # # # # # # #     }
# # # # # # # #     .message {
# # # # # # # #         background-color: #ffffff;
# # # # # # # #         padding: 1rem;
# # # # # # # #         border-radius: 10px;
# # # # # # # #         margin-bottom: 0.5rem;
# # # # # # # #         font-family: cursive; /* Change font family to cursive */
# # # # # # # #     }
# # # # # # # #     .bot {
# # # # # # # #         text-align: left;
# # # # # # # #         color: #2c3e50;
# # # # # # # #     }
# # # # # # # #     .user {
# # # # # # # #         text-align: right;
# # # # # # # #         color: #2980b9;
# # # # # # # #     }
# # # # # # # #     .button {
# # # # # # # #         background-color: #3498db;
# # # # # # # #         color: white;
# # # # # # # #         border: none;
# # # # # # # #         padding: 1rem 2rem;
# # # # # # # #         font-size: 1.2rem;
# # # # # # # #         border-radius: 5px;
# # # # # # # #         cursor: pointer;
# # # # # # # #     }
# # # # # # # #     .button:hover {
# # # # # # # #         background-color: #2980b9;
# # # # # # # #     }
# # # # # # # #     </style>
# # # # # # # # """, unsafe_allow_html=True)

# # # # # # # # # Streamlit app layout
# # # # # # # # st.write("""
# # # # # # # # ### Stress Detection Chatbot
# # # # # # # # Hi there! I'm here to help you detect stress. Go ahead and type your message below.
# # # # # # # # """)

# # # # # # # # # Chat interaction
# # # # # # # # user_input = st.text_input("You:", "")
# # # # # # # # if st.button("Send"):
# # # # # # # #     if user_input:
# # # # # # # #         cleaned_input = clean(user_input)
# # # # # # # #         vectorized_input = tfidf.transform([cleaned_input]).toarray()
# # # # # # # #         prediction = model.predict(vectorized_input)
# # # # # # # #         result = "Stress Detected" if prediction[0] == 1 else "No Stress Detected"

# # # # # # # #         # Update chat history file
# # # # # # # #         chat_history_df = pd.read_csv(CHAT_HISTORY_FILE)
# # # # # # # #         chat_history_df = pd.concat([chat_history_df, pd.DataFrame({'User': [user_input], 'Bot': [result]})], ignore_index=True)
# # # # # # # #         chat_history_df.to_csv(CHAT_HISTORY_FILE, index=False)

# # # # # # # # # Display chat history
# # # # # # # # st.sidebar.title("Chat History")
# # # # # # # # chat_history_df = pd.read_csv(CHAT_HISTORY_FILE)
# # # # # # # # for index, row in chat_history_df.iterrows():
# # # # # # # #     st.sidebar.markdown('<div class="message-container">', unsafe_allow_html=True)
# # # # # # # #     st.sidebar.markdown('<div class="message user">' + row['User'] + '</div>', unsafe_allow_html=True)
# # # # # # # #     st.sidebar.markdown('<div class="message bot">' + row['Bot'] + '</div>', unsafe_allow_html=True)
# # # # # # # #     st.sidebar.markdown('</div>', unsafe_allow_html=True)

# # # # # # # # # Display chat history graph
# # # # # # # # st.sidebar.title("Chat History Graph")
# # # # # # # # if not chat_history_df.empty:
# # # # # # # #     chat_history_df['Bot'] = chat_history_df['Bot'].apply(lambda x: 1 if x == "Stress Detected" else 0)
# # # # # # # #     plt.figure(figsize=(6, 4))
# # # # # # # #     sns.lineplot(data=chat_history_df, x=chat_history_df.index, y='Bot', markers=True)
# # # # # # # #     plt.xlabel('Interaction')
# # # # # # # #     plt.ylabel('Stress (1) / No Stress (0)')
# # # # # # # #     plt.title('Chat History: Stress Detection Over Time')
# # # # # # # #     st.sidebar.pyplot(plt)

# # # # # # # # # Display word cloud
# # # # # # # # st.write("### Word Cloud of the Text Data")
# # # # # # # # wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['text']))
# # # # # # # # plt.figure(figsize=(10, 5))
# # # # # # # # plt.imshow(wordcloud, interpolation='bilinear')
# # # # # # # # plt.axis('off')
# # # # # # # # st.pyplot(plt)

# # # # # # # # # Additional graphs in the main area
# # # # # # # # st.write("### Distribution of Stress and Non-Stress Predictions")
# # # # # # # # if not chat_history_df.empty:
# # # # # # # #     plt.figure(figsize=(6, 4))
# # # # # # # #     sns.countplot(x='Bot', data=chat_history_df, palette='viridis')
# # # # # # # #     plt.xlabel('Prediction')
# # # # # # # #     plt.ylabel('Count')
# # # # # # # #     plt.title('Count of Stress vs Non-Stress Predictions')
# # # # # # # #     st.pyplot(plt)

# # # # # # # #     st.write("### Proportion of Stress vs Non-Stress Predictions")
# # # # # # # #     pie_data = chat_history_df['Bot'].value_counts()
# # # # # # # #     plt.figure(figsize=(6, 6))
# # # # # # # #     plt.pie(pie_data, labels=['No Stress', 'Stress'], autopct='%1.1f%%', startangle=140, colors=['#3498db', '#e74c3c'])
# # # # # # # #     plt.title('Proportion of Stress vs Non-Stress Predictions')
# # # # # # # #     st.pyplot(plt)

# # # # # # # # # Histogram of Message Lengths
# # # # # # # # st.write("### Histogram of Message Lengths")
# # # # # # # # message_lengths = chat_history_df['User'].apply(len)
# # # # # # # # plt.figure(figsize=(6, 4))
# # # # # # # # sns.histplot(message_lengths, bins=30, kde=True, color='purple')
# # # # # # # # plt.xlabel('Message Length')
# # # # # # # # plt.ylabel('Frequency')
# # # # # # # # plt.title('Distribution of Message Lengths')
# # # # # # # # st.pyplot(plt)

# # # # # # # # # Confusion Matrix Heatmap
# # # # # # # # st.write("### Confusion Matrix")
# # # # # # # # conf_matrix = confusion_matrix(y_test, y_pred)
# # # # # # # # plt.figure(figsize=(6, 4))
# # # # # # # # sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Stress', 'Stress'], yticklabels=['No Stress', 'Stress'])
# # # # # # # # plt.xlabel('Predicted')
# # # # # # # # plt.ylabel('Actual')
# # # # # # # # plt.title('Confusion Matrix')
# # # # # # # # st.pyplot(plt)

# # # # # # # # # Add some information about the app
# # # # # # # # st.write("""
# # # # # # # # #### How it works:
# # # # # # # # - This chatbot uses a Naive Bayes classifier trained on textual data to detect stress.
# # # # # # # # - Type any message above and click on 'Send' to see if stress is detected.
# # # # # # # # - The model preprocesses the text by cleaning it and converting it into numerical features using TF-IDF vectorization.
# # # # # # # # """)

# # # # # # # # # Display performance metrics
# # # # # # # # st.write("### Model Performance Metrics")
# # # # # # # # st.write(f"Accuracy: {accuracy:.2f}")
# # # # # # # # st.write(f"Precision: {precision:.2f}")
# # # # # # # # st.write(f"Recall: {recall:.2f}")
# # # # # # # # st.write(f"F1 Score: {f1:.2f}")

# # # # # # # # # Footer
# # # # # # # # st.markdown("""
# # # # # # # #     <div style='text-align: center; padding: 1rem; font-size: 0.9rem; color: #7f8c8d;'>
# # # # # # # #     Developed by Hassan Arzoo - Roll No: CSE204008, Guided by Dr. Abhishek Das, Associate Professor, CSE Department, Aliah University
# # # # # # # #     </div>
# # # # # # # # """, unsafe_allow_html=True)



# # # # # # # import streamlit as st
# # # # # # # import pandas as pd
# # # # # # # import numpy as np
# # # # # # # from sklearn.feature_extraction.text import TfidfVectorizer
# # # # # # # from sklearn.naive_bayes import MultinomialNB
# # # # # # # from sklearn.model_selection import train_test_split
# # # # # # # from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# # # # # # # import nltk
# # # # # # # import re
# # # # # # # from nltk.corpus import stopwords
# # # # # # # import string
# # # # # # # from wordcloud import WordCloud
# # # # # # # import matplotlib.pyplot as plt
# # # # # # # import seaborn as sns
# # # # # # # import os

# # # # # # # # Download stopwords
# # # # # # # nltk.download('stopwords')

# # # # # # # # Load data
# # # # # # # df = pd.read_csv('stress.csv')

# # # # # # # # Preprocess data
# # # # # # # stemmer = nltk.SnowballStemmer("english")
# # # # # # # stopword = set(stopwords.words('english'))

# # # # # # # def clean(text):
# # # # # # #     text = str(text).lower()
# # # # # # #     text = re.sub(r'\[.*?\]', '', text)
# # # # # # #     text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
# # # # # # #     text = re.sub(r'\w*\d\w*', '', text)
# # # # # # #     text = re.sub(r'[''""…]', '', text)
# # # # # # #     text = re.sub(r'\n', '', text)
# # # # # # #     text = [word for word in text.split(' ') if word not in stopword]
# # # # # # #     text = ' '.join(text)
# # # # # # #     text = [stemmer.stem(word) for word in text.split(' ')]
# # # # # # #     text = ' '.join(text)
# # # # # # #     return text

# # # # # # # df['text'] = df['text'].apply(clean)

# # # # # # # # Feature extraction
# # # # # # # tfidf = TfidfVectorizer(max_features=5000)
# # # # # # # X = tfidf.fit_transform(df['text']).toarray()
# # # # # # # y = df['label']

# # # # # # # # Train model
# # # # # # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # # # # # # model = MultinomialNB()
# # # # # # # model.fit(X_train, y_train)
# # # # # # # y_pred = model.predict(X_test)

# # # # # # # # Performance metrics
# # # # # # # accuracy = accuracy_score(y_test, y_pred)
# # # # # # # precision = precision_score(y_test, y_pred)
# # # # # # # recall = recall_score(y_test, y_pred)
# # # # # # # f1 = f1_score(y_test, y_pred)

# # # # # # # # Initialize chat history file
# # # # # # # CHAT_HISTORY_FILE = 'chat_history.csv'
# # # # # # # if not os.path.exists(CHAT_HISTORY_FILE):
# # # # # # #     pd.DataFrame(columns=['User', 'Bot']).to_csv(CHAT_HISTORY_FILE, index=False)

# # # # # # # # Custom CSS for styling
# # # # # # # st.markdown("""
# # # # # # #     <style>
# # # # # # #     .message-container {
# # # # # # #         margin-bottom: 1rem;
# # # # # # #     }
# # # # # # #     .message {
# # # # # # #         background-color: #ffffff;
# # # # # # #         padding: 1rem;
# # # # # # #         border-radius: 10px;
# # # # # # #         margin-bottom: 0.5rem;
# # # # # # #         font-family: cursive; /* Change font family to cursive */
# # # # # # #     }
# # # # # # #     .bot {
# # # # # # #         text-align: left;
# # # # # # #         color: #2c3e50;
# # # # # # #     }
# # # # # # #     .user {
# # # # # # #         text-align: right;
# # # # # # #         color: #2980b9;
# # # # # # #     }
# # # # # # #     .button {
# # # # # # #         background-color: #3498db;
# # # # # # #         color: white;
# # # # # # #         border: none;
# # # # # # #         padding: 1rem 2rem;
# # # # # # #         font-size: 1.2rem;
# # # # # # #         border-radius: 5px;
# # # # # # #         cursor: pointer;
# # # # # # #     }
# # # # # # #     .button:hover {
# # # # # # #         background-color: #2980b9;
# # # # # # #     }
# # # # # # #     </style>
# # # # # # # """, unsafe_allow_html=True)

# # # # # # # # Streamlit app layout
# # # # # # # st.write("""
# # # # # # # ### Stress Detection Chatbot
# # # # # # # Hi there! I'm here to help you detect stress. Go ahead and type your message below.
# # # # # # # """)

# # # # # # # # Chat interaction
# # # # # # # user_input = st.text_input("You:", "")
# # # # # # # if st.button("Send"):
# # # # # # #     if user_input:
# # # # # # #         cleaned_input = clean(user_input)
# # # # # # #         vectorized_input = tfidf.transform([cleaned_input]).toarray()
# # # # # # #         prediction = model.predict(vectorized_input)
# # # # # # #         result = "Stress Detected" if prediction[0] == 1 else "No Stress Detected"

# # # # # # #         # Update chat history file
# # # # # # #         chat_history_df = pd.read_csv(CHAT_HISTORY_FILE)
# # # # # # #         chat_history_df = pd.concat([chat_history_df, pd.DataFrame({'User': [user_input], 'Bot': [result]})], ignore_index=True)
# # # # # # #         chat_history_df.to_csv(CHAT_HISTORY_FILE, index=False)

# # # # # # #         st.write(f"**Bot:** {result}")

# # # # # # # # Display chat history
# # # # # # # st.sidebar.title("Chat History")
# # # # # # # chat_history_df = pd.read_csv(CHAT_HISTORY_FILE)
# # # # # # # for index, row in chat_history_df.iterrows():
# # # # # # #     st.sidebar.markdown('<div class="message-container">', unsafe_allow_html=True)
# # # # # # #     st.sidebar.markdown('<div class="message user">' + row['User'] + '</div>', unsafe_allow_html=True)
# # # # # # #     st.sidebar.markdown('<div class="message bot">' + row['Bot'] + '</div>', unsafe_allow_html=True)
# # # # # # #     st.sidebar.markdown('</div>', unsafe_allow_html=True)

# # # # # # # # Display chat history graph
# # # # # # # st.sidebar.title("Chat History Graph")
# # # # # # # if not chat_history_df.empty:
# # # # # # #     chat_history_df['Bot'] = chat_history_df['Bot'].apply(lambda x: 1 if x == "Stress Detected" else 0)
# # # # # # #     plt.figure(figsize=(6, 4))
# # # # # # #     sns.lineplot(data=chat_history_df, x=chat_history_df.index, y='Bot', markers=True)
# # # # # # #     plt.xlabel('Interaction')
# # # # # # #     plt.ylabel('Stress (1) / No Stress (0)')
# # # # # # #     plt.title('Chat History: Stress Detection Over Time')
# # # # # # #     st.sidebar.pyplot(plt)

# # # # # # # # Display word cloud
# # # # # # # st.write("### Word Cloud of the Text Data")
# # # # # # # wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['text']))
# # # # # # # plt.figure(figsize=(10, 5))
# # # # # # # plt.imshow(wordcloud, interpolation='bilinear')
# # # # # # # plt.axis('off')
# # # # # # # st.pyplot(plt)

# # # # # # # # Additional graphs in the main area
# # # # # # # st.write("### Distribution of Stress and Non-Stress Predictions")
# # # # # # # if not chat_history_df.empty:
# # # # # # #     plt.figure(figsize=(6, 4))
# # # # # # #     sns.countplot(x='Bot', data=chat_history_df, palette='viridis')
# # # # # # #     plt.xlabel('Prediction')
# # # # # # #     plt.ylabel('Count')
# # # # # # #     plt.title('Count of Stress vs Non-Stress Predictions')
# # # # # # #     st.pyplot(plt)

# # # # # # #     st.write("### Proportion of Stress vs Non-Stress Predictions")
# # # # # # #     pie_data = chat_history_df['Bot'].value_counts()
# # # # # # #     plt.figure(figsize=(6, 6))
# # # # # # #     plt.pie(pie_data, labels=['No Stress', 'Stress'], autopct='%1.1f%%', startangle=140, colors=['#3498db', '#e74c3c'])
# # # # # # #     plt.title('Proportion of Stress vs Non-Stress Predictions')
# # # # # # #     st.pyplot(plt)

# # # # # # # # Histogram of Message Lengths
# # # # # # # st.write("### Histogram of Message Lengths")
# # # # # # # message_lengths = chat_history_df['User'].apply(len)
# # # # # # # plt.figure(figsize=(6, 4))
# # # # # # # sns.histplot(message_lengths, bins=30, kde=True, color='purple')
# # # # # # # plt.xlabel('Message Length')
# # # # # # # plt.ylabel('Frequency')
# # # # # # # plt.title('Distribution of Message Lengths')
# # # # # # # st.pyplot(plt)

# # # # # # # # Confusion Matrix Heatmap
# # # # # # # st.write("### Confusion Matrix")
# # # # # # # conf_matrix = confusion_matrix(y_test, y_pred)
# # # # # # # plt.figure(figsize=(6, 4))
# # # # # # # sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Stress', 'Stress'], yticklabels=['No Stress', 'Stress'])
# # # # # # # plt.xlabel('Predicted')
# # # # # # # plt.ylabel('Actual')
# # # # # # # plt.title('Confusion Matrix')
# # # # # # # st.pyplot(plt)

# # # # # # # # Add some information about the app
# # # # # # # st.write("""
# # # # # # # #### How it works:
# # # # # # # - This chatbot uses a Naive Bayes classifier trained on textual data to detect stress.
# # # # # # # - Type any message above and click on 'Send' to see if stress is detected.
# # # # # # # - The model preprocesses the text by cleaning it and converting it into numerical features using TF-IDF vectorization.
# # # # # # # """)

# # # # # # # # Display performance metrics
# # # # # # # st.write("### Model Performance Metrics")
# # # # # # # st.write(f"Accuracy: {accuracy:.2f}")
# # # # # # # st.write(f"Precision: {precision:.2f}")
# # # # # # # st.write(f"Recall: {recall:.2f}")
# # # # # # # st.write(f"F1 Score: {f1:.2f}")

# # # # # # # # Footer
# # # # # # # st.markdown("""
# # # # # # #     <div style='text-align: center; padding: 1rem; font-size: 0.9rem; color: #7f8c8d;'>
# # # # # # #     Developed by Hassan Arzoo - Roll No: CSE204008, Guided by Dr. Abhishek Das, Associate Professor, CSE Department, Aliah University
# # # # # # #     </div>
# # # # # # # """, unsafe_allow_html=True)


# # # # # # import streamlit as st
# # # # # # import pandas as pd
# # # # # # import numpy as np
# # # # # # from sklearn.feature_extraction.text import TfidfVectorizer
# # # # # # from sklearn.naive_bayes import MultinomialNB
# # # # # # from sklearn.model_selection import train_test_split
# # # # # # from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# # # # # # import nltk
# # # # # # import re
# # # # # # from nltk.corpus import stopwords
# # # # # # import string
# # # # # # from wordcloud import WordCloud
# # # # # # import matplotlib.pyplot as plt
# # # # # # import seaborn as sns
# # # # # # import os
# # # # # # import time

# # # # # # # Download stopwords
# # # # # # nltk.download('stopwords')

# # # # # # # Load data
# # # # # # df = pd.read_csv('stress.csv')

# # # # # # # Preprocess data
# # # # # # stemmer = nltk.SnowballStemmer("english")
# # # # # # stopword = set(stopwords.words('english'))

# # # # # # def clean(text):
# # # # # #     text = str(text).lower()
# # # # # #     text = re.sub(r'\[.*?\]', '', text)
# # # # # #     text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
# # # # # #     text = re.sub(r'\w*\d\w*', '', text)
# # # # # #     text = re.sub(r'[''""…]', '', text)
# # # # # #     text = re.sub(r'\n', '', text)
# # # # # #     text = [word for word in text.split(' ') if word not in stopword]
# # # # # #     text = ' '.join(text)
# # # # # #     text = [stemmer.stem(word) for word in text.split(' ')]
# # # # # #     text = ' '.join(text)
# # # # # #     return text

# # # # # # df['text'] = df['text'].apply(clean)

# # # # # # # Feature extraction
# # # # # # tfidf = TfidfVectorizer(max_features=5000)
# # # # # # X = tfidf.fit_transform(df['text']).toarray()
# # # # # # y = df['label']

# # # # # # # Train model
# # # # # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # # # # # model = MultinomialNB()
# # # # # # model.fit(X_train, y_train)
# # # # # # y_pred = model.predict(X_test)

# # # # # # # Performance metrics
# # # # # # accuracy = accuracy_score(y_test, y_pred)
# # # # # # precision = precision_score(y_test, y_pred)
# # # # # # recall = recall_score(y_test, y_pred)
# # # # # # f1 = f1_score(y_test, y_pred)

# # # # # # # Initialize chat history file
# # # # # # CHAT_HISTORY_FILE = 'chat_history.csv'
# # # # # # if not os.path.exists(CHAT_HISTORY_FILE):
# # # # # #     pd.DataFrame(columns=['User', 'Bot']).to_csv(CHAT_HISTORY_FILE, index=False)

# # # # # # # Custom CSS for styling
# # # # # # st.markdown("""
# # # # # #     <style>
# # # # # #     .message-container {
# # # # # #         margin-bottom: 1rem;
# # # # # #     }
# # # # # #     .message {
# # # # # #         background-color: #ffffff;
# # # # # #         padding: 1rem;
# # # # # #         border-radius: 10px;
# # # # # #         margin-bottom: 0.5rem;
# # # # # #         font-family: cursive; /* Change font family to cursive */
# # # # # #     }
# # # # # #     .bot {
# # # # # #         text-align: left;
# # # # # #         color: #2c3e50;
# # # # # #     }
# # # # # #     .user {
# # # # # #         text-align: right;
# # # # # #         color: #2980b9;
# # # # # #     }
# # # # # #     .button {
# # # # # #         background-color: #3498db;
# # # # # #         color: white;
# # # # # #         border: none;
# # # # # #         padding: 1rem 2rem;
# # # # # #         font-size: 1.2rem;
# # # # # #         border-radius: 5px;
# # # # # #         cursor: pointer;
# # # # # #     }
# # # # # #     .button:hover {
# # # # # #         background-color: #2980b9;
# # # # # #     }
# # # # # #     </style>
# # # # # # """, unsafe_allow_html=True)

# # # # # # # Streamlit app layout
# # # # # # st.write("""
# # # # # # ### Stress Detection Chatbot
# # # # # # Hi there! I'm here to help you detect stress. Go ahead and type your message below.
# # # # # # """)

# # # # # # # Chat interaction
# # # # # # user_input = st.text_input("You:", "")
# # # # # # if st.button("Send"):
# # # # # #     if user_input:
# # # # # #         with st.spinner('Calculating...'):
# # # # # #             time.sleep(1)  # Simulate a delay
# # # # # #             cleaned_input = clean(user_input)
# # # # # #             vectorized_input = tfidf.transform([cleaned_input]).toarray()
# # # # # #             prediction = model.predict(vectorized_input)
# # # # # #             result = "**Stress Detected**" if prediction[0] == 1 else "**No Stress Detected**"

# # # # # #             # Update chat history file
# # # # # #             chat_history_df = pd.read_csv(CHAT_HISTORY_FILE)
# # # # # #             chat_history_df = pd.concat([chat_history_df, pd.DataFrame({'User': [user_input], 'Bot': [result]})], ignore_index=True)
# # # # # #             chat_history_df.to_csv(CHAT_HISTORY_FILE, index=False)

# # # # # #             st.write(f"**Bot:** {result}")

# # # # # # # Display chat history
# # # # # # st.sidebar.title("Chat History")
# # # # # # chat_history_df = pd.read_csv(CHAT_HISTORY_FILE)
# # # # # # for index, row in chat_history_df.iterrows():
# # # # # #     st.sidebar.markdown('<div class="message-container">', unsafe_allow_html=True)
# # # # # #     st.sidebar.markdown('<div class="message user">' + row['User'] + '</div>', unsafe_allow_html=True)
# # # # # #     st.sidebar.markdown('<div class="message bot">' + row['Bot'] + '</div>', unsafe_allow_html=True)
# # # # # #     st.sidebar.markdown('</div>', unsafe_allow_html=True)

# # # # # # # Display chat history graph
# # # # # # st.sidebar.title("Chat History Graph")
# # # # # # if not chat_history_df.empty:
# # # # # #     chat_history_df['Bot'] = chat_history_df['Bot'].apply(lambda x: 1 if x == "**Stress Detected**" else 0)
# # # # # #     plt.figure(figsize=(6, 4))
# # # # # #     sns.lineplot(data=chat_history_df, x=chat_history_df.index, y='Bot', markers=True)
# # # # # #     plt.xlabel('Interaction')
# # # # # #     plt.ylabel('Stress (1) / No Stress (0)')
# # # # # #     plt.title('Chat History: Stress Detection Over Time')
# # # # # #     st.sidebar.pyplot(plt)

# # # # # # # Display word cloud
# # # # # # st.write("### Word Cloud of the Text Data")
# # # # # # wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['text']))
# # # # # # plt.figure(figsize=(10, 5))
# # # # # # plt.imshow(wordcloud, interpolation='bilinear')
# # # # # # plt.axis('off')
# # # # # # st.pyplot(plt)

# # # # # # # Additional graphs in the main area
# # # # # # st.write("### Distribution of Stress and Non-Stress Predictions")
# # # # # # if not chat_history_df.empty:
# # # # # #     plt.figure(figsize=(6, 4))
# # # # # #     sns.countplot(x='Bot', data=chat_history_df, palette='viridis')
# # # # # #     plt.xlabel('Prediction')
# # # # # #     plt.ylabel('Count')
# # # # # #     plt.title('Count of Stress vs Non-Stress Predictions')
# # # # # #     st.pyplot(plt)

# # # # # #     st.write("### Proportion of Stress vs Non-Stress Predictions")
# # # # # #     pie_data = chat_history_df['Bot'].value_counts()
# # # # # #     plt.figure(figsize=(6, 6))
# # # # # #     plt.pie(pie_data, labels=['No Stress', 'Stress'], autopct='%1.1f%%', startangle=140, colors=['#3498db', '#e74c3c'])
# # # # # #     plt.title('Proportion of Stress vs Non-Stress Predictions')
# # # # # #     st.pyplot(plt)

# # # # # # # Histogram of Message Lengths
# # # # # # st.write("### Histogram of Message Lengths")
# # # # # # message_lengths = chat_history_df['User'].apply(len)
# # # # # # plt.figure(figsize=(6, 4))
# # # # # # sns.histplot(message_lengths, bins=30, kde=True, color='purple')
# # # # # # plt.xlabel('Message Length')
# # # # # # plt.ylabel('Frequency')
# # # # # # plt.title('Distribution of Message Lengths')
# # # # # # st.pyplot(plt)

# # # # # # # Confusion Matrix Heatmap
# # # # # # st.write("### Confusion Matrix")
# # # # # # conf_matrix = confusion_matrix(y_test, y_pred)
# # # # # # plt.figure(figsize=(6, 4))
# # # # # # sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Stress', 'Stress'], yticklabels=['No Stress', 'Stress'])
# # # # # # plt.xlabel('Predicted')
# # # # # # plt.ylabel('Actual')
# # # # # # plt.title('Confusion Matrix')
# # # # # # st.pyplot(plt)

# # # # # # # Add some information about the app
# # # # # # st.write("""
# # # # # # #### How it works:
# # # # # # - This chatbot uses a Naive Bayes classifier trained on textual data to detect stress.
# # # # # # - Type any message above and click on 'Send' to see if stress is detected.
# # # # # # - The model preprocesses the text by cleaning it and converting it into numerical features using TF-IDF vectorization.
# # # # # # """)

# # # # # # # Display performance metrics
# # # # # # st.write("### Model Performance Metrics")
# # # # # # st.write(f"Accuracy: {accuracy:.2f}")
# # # # # # st.write(f"Precision: {precision:.2f}")
# # # # # # st.write(f"Recall: {recall:.2f}")
# # # # # # st.write(f"F1 Score: {f1:.2f}")

# # # # # # # Footer
# # # # # # st.markdown("""
# # # # # #     <div style='text-align: center; padding: 1rem; font-size: 0.9rem; color: #7f8c8d;'>
# # # # # #     Developed by Hassan Arzoo - Roll No: CSE204008, Guided by Dr. Abhishek Das, Associate Professor, CSE Department, Aliah University
# # # # # #     </div>
# # # # # # """, unsafe_allow_html=True)

# # # # # import streamlit as st
# # # # # import pandas as pd
# # # # # import numpy as np
# # # # # from sklearn.feature_extraction.text import TfidfVectorizer
# # # # # from sklearn.naive_bayes import MultinomialNB
# # # # # from sklearn.model_selection import train_test_split
# # # # # from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# # # # # import nltk
# # # # # import re
# # # # # from nltk.corpus import stopwords
# # # # # import string
# # # # # from wordcloud import WordCloud
# # # # # import matplotlib.pyplot as plt
# # # # # import seaborn as sns
# # # # # import os
# # # # # import time

# # # # # # Download stopwords
# # # # # nltk.download('stopwords')

# # # # # # Load data
# # # # # df = pd.read_csv('stress.csv')

# # # # # # Preprocess data
# # # # # stemmer = nltk.SnowballStemmer("english")
# # # # # stopword = set(stopwords.words('english'))

# # # # # def clean(text):
# # # # #     text = str(text).lower()
# # # # #     text = re.sub(r'\[.*?\]', '', text)
# # # # #     text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
# # # # #     text = re.sub(r'\w*\d\w*', '', text)
# # # # #     text = re.sub(r'[''""…]', '', text)
# # # # #     text = re.sub(r'\n', '', text)
# # # # #     text = [word for word in text.split(' ') if word not in stopword]
# # # # #     text = ' '.join(text)
# # # # #     text = [stemmer.stem(word) for word in text.split(' ')]
# # # # #     text = ' '.join(text)
# # # # #     return text

# # # # # df['text'] = df['text'].apply(clean)

# # # # # # Feature extraction
# # # # # tfidf = TfidfVectorizer(max_features=5000)
# # # # # X = tfidf.fit_transform(df['text']).toarray()
# # # # # y = df['label']

# # # # # # Train model
# # # # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # # # # model = MultinomialNB()
# # # # # model.fit(X_train, y_train)
# # # # # y_pred = model.predict(X_test)

# # # # # # Performance metrics
# # # # # accuracy = accuracy_score(y_test, y_pred)
# # # # # precision = precision_score(y_test, y_pred)
# # # # # recall = recall_score(y_test, y_pred)
# # # # # f1 = f1_score(y_test, y_pred)

# # # # # # Initialize chat history file
# # # # # CHAT_HISTORY_FILE = 'chat_history.csv'
# # # # # if not os.path.exists(CHAT_HISTORY_FILE):
# # # # #     pd.DataFrame(columns=['User', 'Bot']).to_csv(CHAT_HISTORY_FILE, index=False)

# # # # # # Custom CSS for styling
# # # # # st.markdown("""
# # # # #     <style>
# # # # #     .message-container {
# # # # #         margin-bottom: 1rem;
# # # # #     }
# # # # #     .message {
# # # # #         background-color: #ffffff;
# # # # #         padding: 1rem;
# # # # #         border-radius: 10px;
# # # # #         margin-bottom: 0.5rem;
# # # # #         font-family: cursive; /* Change font family to cursive */
# # # # #     }
# # # # #     .bot {
# # # # #         text-align: left;
# # # # #         color: #2c3e50;
# # # # #     }
# # # # #     .user {
# # # # #         text-align: right;
# # # # #         color: #2980b9;
# # # # #     }
# # # # #     .button {
# # # # #         background-color: #3498db;
# # # # #         color: white;
# # # # #         border: none;
# # # # #         padding: 1rem 2rem;
# # # # #         font-size: 1.2rem;
# # # # #         border-radius: 5px;
# # # # #         cursor: pointer;
# # # # #     }
# # # # #     .button:hover {
# # # # #         background-color: #2980b9;
# # # # #     }
# # # # #     </style>
# # # # # """, unsafe_allow_html=True)

# # # # # # Streamlit app layout
# # # # # st.write("""
# # # # # ### Stress Detection Chatbot
# # # # # Hi there! I'm here to help you detect stress. Go ahead and type your message below.
# # # # # """)

# # # # # # Chat interaction
# # # # # user_input = st.text_input("You:", "")
# # # # # if st.button("Send"):
# # # # #     if user_input:
# # # # #         with st.spinner('Calculating...'):
# # # # #             time.sleep(1)  # Simulate a delay
# # # # #             cleaned_input = clean(user_input)
# # # # #             vectorized_input = tfidf.transform([cleaned_input]).toarray()
# # # # #             prediction = model.predict(vectorized_input)
# # # # #             result = "**<b>Stress Detected</b>**" if prediction[0] == 1 else "**<b>No Stress Detected</b>**"

# # # # #             # Update chat history file
# # # # #             chat_history_df = pd.read_csv(CHAT_HISTORY_FILE)
# # # # #             chat_history_df = pd.concat([chat_history_df, pd.DataFrame({'User': [user_input], 'Bot': [result]})], ignore_index=True)
# # # # #             chat_history_df.to_csv(CHAT_HISTORY_FILE, index=False)

# # # # #             st.markdown(f"**Bot:** {result}", unsafe_allow_html=True)

# # # # # # Display chat history
# # # # # st.sidebar.title("Chat History")
# # # # # chat_history_df = pd.read_csv(CHAT_HISTORY_FILE)
# # # # # for index, row in chat_history_df.iterrows():
# # # # #     st.sidebar.markdown('<div class="message-container">', unsafe_allow_html=True)
# # # # #     st.sidebar.markdown('<div class="message user">' + row['User'] + '</div>', unsafe_allow_html=True)
# # # # #     st.sidebar.markdown('<div class="message bot">' + row['Bot'] + '</div>', unsafe_allow_html=True)
# # # # #     st.sidebar.markdown('</div>', unsafe_allow_html=True)

# # # # # # Display chat history graph
# # # # # st.sidebar.title("Chat History Graph")
# # # # # if not chat_history_df.empty:
# # # # #     chat_history_df['Bot'] = chat_history_df['Bot'].apply(lambda x: 1 if x == "**<b>Stress Detected</b>**" else 0)
# # # # #     plt.figure(figsize=(6, 4))
# # # # #     sns.lineplot(data=chat_history_df, x=chat_history_df.index, y='Bot', markers=True)
# # # # #     plt.xlabel('Interaction')
# # # # #     plt.ylabel('Stress (1) / No Stress (0)')
# # # # #     plt.title('Chat History: Stress Detection Over Time')
# # # # #     st.sidebar.pyplot(plt)

# # # # # # Display word cloud
# # # # # st.write("### Word Cloud of the Text Data")
# # # # # wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['text']))
# # # # # plt.figure(figsize=(10, 5))
# # # # # plt.imshow(wordcloud, interpolation='bilinear')
# # # # # plt.axis('off')
# # # # # st.pyplot(plt)

# # # # # # Additional graphs in the main area
# # # # # st.write("### Distribution of Stress and Non-Stress Predictions")
# # # # # if not chat_history_df.empty:
# # # # #     plt.figure(figsize=(6, 4))
# # # # #     sns.countplot(x='Bot', data=chat_history_df, palette='viridis')
# # # # #     plt.xlabel('Prediction')
# # # # #     plt.ylabel('Count')
# # # # #     plt.title('Count of Stress vs Non-Stress Predictions')
# # # # #     st.pyplot(plt)

# # # # #     st.write("### Proportion of Stress vs Non-Stress Predictions")
# # # # #     pie_data = chat_history_df['Bot'].value_counts()
# # # # #     plt.figure(figsize=(6, 6))
# # # # #     plt.pie(pie_data, labels=['No Stress', 'Stress'], autopct='%1.1f%%', startangle=140, colors=['#3498db', '#e74c3c'])
# # # # #     plt.title('Proportion of Stress vs Non-Stress Predictions')
# # # # #     st.pyplot(plt)

# # # # # # Histogram of Message Lengths
# # # # # st.write("### Histogram of Message Lengths")
# # # # # message_lengths = chat_history_df['User'].apply(len)
# # # # # plt.figure(figsize=(6, 4))
# # # # # sns.histplot(message_lengths, bins=30, kde=True, color='purple')
# # # # # plt.xlabel('Message Length')
# # # # # plt.ylabel('Frequency')
# # # # # plt.title('Distribution of Message Lengths')
# # # # # st.pyplot(plt)

# # # # # # Confusion Matrix Heatmap
# # # # # st.write("### Confusion Matrix")
# # # # # conf_matrix = confusion_matrix(y_test, y_pred)
# # # # # plt.figure(figsize=(6, 4))
# # # # # sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Stress', 'Stress'], yticklabels=['No Stress', 'Stress'])
# # # # # plt.xlabel('Predicted')
# # # # # plt.ylabel('Actual')
# # # # # plt.title('Confusion Matrix')
# # # # # st.pyplot(plt)

# # # # # # Add some information about the app
# # # # # st.write("""
# # # # # #### How it works:
# # # # # - This chatbot uses a Naive Bayes classifier trained on textual data to detect stress.
# # # # # - Type any message above and click on 'Send' to see if stress is detected.
# # # # # - The model preprocesses the text by cleaning it and converting it into numerical features using TF-IDF vectorization.
# # # # # """)

# # # # # # Display performance metrics
# # # # # st.write("### Model Performance Metrics")
# # # # # st.write(f"Accuracy: {accuracy:.2f}")
# # # # # st.write(f"Precision: {precision:.2f}")
# # # # # st.write(f"Recall: {recall:.2f}")
# # # # # st.write(f"F1 Score: {f1:.2f}")

# # # # # # Footer
# # # # # st.markdown("""
# # # # #     <div style='text-align: center; padding: 1rem; font-size: 0.9rem; color: #7f8c8d;'>
# # # # #     Developed by Hassan Arzoo - Roll No: CSE204008, Guided by Dr. Abhishek Das, Associate Professor, CSE Department, Aliah University
# # # # #     </div>
# # # # # """, unsafe_allow_html=True)


# # # # import streamlit as st
# # # # import pandas as pd
# # # # import numpy as np
# # # # from sklearn.feature_extraction.text import TfidfVectorizer
# # # # from sklearn.naive_bayes import MultinomialNB
# # # # from sklearn.model_selection import train_test_split
# # # # from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# # # # import nltk
# # # # import re
# # # # from nltk.corpus import stopwords
# # # # import string
# # # # from wordcloud import WordCloud
# # # # import matplotlib.pyplot as plt
# # # # import seaborn as sns
# # # # import os
# # # # import time

# # # # # Download stopwords
# # # # nltk.download('stopwords')

# # # # # Load data
# # # # df = pd.read_csv('stress.csv')

# # # # # Preprocess data
# # # # stemmer = nltk.SnowballStemmer("english")
# # # # stopword = set(stopwords.words('english'))

# # # # def clean(text):
# # # #     text = str(text).lower()
# # # #     text = re.sub(r'\[.*?\]', '', text)
# # # #     text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
# # # #     text = re.sub(r'\w*\d\w*', '', text)
# # # #     text = re.sub(r'[''""…]', '', text)
# # # #     text = re.sub(r'\n', '', text)
# # # #     text = [word for word in text.split(' ') if word not in stopword]
# # # #     text = ' '.join(text)
# # # #     text = [stemmer.stem(word) for word in text.split(' ')]
# # # #     text = ' '.join(text)
# # # #     return text

# # # # df['text'] = df['text'].apply(clean)

# # # # # Feature extraction
# # # # tfidf = TfidfVectorizer(max_features=5000)
# # # # X = tfidf.fit_transform(df['text']).toarray()
# # # # y = df['label']

# # # # # Train model
# # # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # # # model = MultinomialNB()
# # # # model.fit(X_train, y_train)
# # # # y_pred = model.predict(X_test)

# # # # # Performance metrics
# # # # accuracy = accuracy_score(y_test, y_pred)
# # # # precision = precision_score(y_test, y_pred)
# # # # recall = recall_score(y_test, y_pred)
# # # # f1 = f1_score(y_test, y_pred)

# # # # # Initialize chat history file
# # # # CHAT_HISTORY_FILE = 'chat_history.csv'
# # # # if not os.path.exists(CHAT_HISTORY_FILE):
# # # #     pd.DataFrame(columns=['User', 'Bot']).to_csv(CHAT_HISTORY_FILE, index=False)

# # # # # Custom CSS for styling
# # # # st.markdown("""
# # # #     <style>
# # # #     .message-container {
# # # #         margin-bottom: 1rem;
# # # #     }
# # # #     .message {
# # # #         background-color: #ffffff;
# # # #         padding: 1rem;
# # # #         border-radius: 10px;
# # # #         margin-bottom: 0.5rem;
# # # #         font-family: cursive; /* Change font family to cursive */
# # # #     }
# # # #     .bot {
# # # #         text-align: left;
# # # #         color: #2c3e50;
# # # #     }
# # # #     .user {
# # # #         text-align: right;
# # # #         color: #2980b9;
# # # #     }
# # # #     .bold {
# # # #         font-weight: bold;
# # # #     }
# # # #     </style>
# # # # """, unsafe_allow_html=True)

# # # # # Streamlit app layout
# # # # st.write("""
# # # # ### Stress Detection Chatbot
# # # # Hi there! I'm here to help you detect stress. Go ahead and type your message below.
# # # # """)

# # # # # Chat interaction
# # # # user_input = st.text_input("You:", "")
# # # # if st.button("Send"):
# # # #     if user_input:
# # # #         with st.spinner('Calculating...'):
# # # #             time.sleep(1)  # Simulate a delay
# # # #             cleaned_input = clean(user_input)
# # # #             vectorized_input = tfidf.transform([cleaned_input]).toarray()
# # # #             prediction = model.predict(vectorized_input)
# # # #             result = "<span class='bold'>Stress Detected</span>" if prediction[0] == 1 else "<span class='bold'>No Stress Detected</span>"

# # # #             # Update chat history file
# # # #             chat_history_df = pd.read_csv(CHAT_HISTORY_FILE)
# # # #             chat_history_df = pd.concat([chat_history_df, pd.DataFrame({'User': [user_input], 'Bot': [result]})], ignore_index=True)
# # # #             chat_history_df.to_csv(CHAT_HISTORY_FILE, index=False)

# # # #             # Display chat messages
# # # #             st.markdown(f"<div class='message user'>{user_input}</div>", unsafe_allow_html=True)
# # # #             st.markdown(f"<div class='message bot'>{result}</div>", unsafe_allow_html=True)

# # # # # Display chat history
# # # # st.sidebar.title("Chat History")
# # # # chat_history_df = pd.read_csv(CHAT_HISTORY_FILE)
# # # # for index, row in chat_history_df.iterrows():
# # # #     st.sidebar.markdown('<div class="message-container">', unsafe_allow_html=True)
# # # #     st.sidebar.markdown('<div class="message user">' + row['User'] + '</div>', unsafe_allow_html=True)
# # # #     st.sidebar.markdown('<div class="message bot">' + row['Bot'] + '</div>', unsafe_allow_html=True)
# # # #     st.sidebar.markdown('</div>', unsafe_allow_html=True)

# # # # # Display chat history graph
# # # # st.sidebar.title("Chat History Graph")
# # # # if not chat_history_df.empty:
# # # #     chat_history_df['Bot'] = chat_history_df['Bot'].apply(lambda x: 1 if x == "<span class='bold'>Stress Detected</span>" else 0)
# # # #     plt.figure(figsize=(6, 4))
# # # #     sns.lineplot(data=chat_history_df, x=chat_history_df.index, y='Bot', markers=True)
# # # #     plt.xlabel('Interaction')
# # # #     plt.ylabel('Stress (1) / No Stress (0)')
# # # #     plt.title('Chat History: Stress Detection Over Time')
# # # #     st.sidebar.pyplot(plt)

# # # # # Display word cloud
# # # # st.write("### Word Cloud of the Text Data")
# # # # wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['text']))
# # # # plt.figure(figsize=(10, 5))
# # # # plt.imshow(wordcloud, interpolation='bilinear')
# # # # plt.axis('off')
# # # # st.pyplot(plt)

# # # # # Additional graphs in the main area
# # # # st.write("### Distribution of Stress and Non-Stress Predictions")
# # # # if not chat_history_df.empty:
# # # #     plt.figure(figsize=(6, 4))
# # # #     sns.countplot(x='Bot', data=chat_history_df, palette='viridis')
# # # #     plt.xlabel('Prediction')
# # # #     plt.ylabel('Count')
# # # #     plt.title('Count of Stress vs Non-Stress Predictions')
# # # #     st.pyplot(plt)

# # # #     st.write("### Proportion of Stress vs Non-Stress Predictions")
# # # #     pie_data = chat_history_df['Bot'].value_counts()
# # # #     plt.figure(figsize=(6, 6))
# # # #     plt.pie(pie_data, labels=['No Stress', 'Stress'], autopct='%1.1f%%', startangle=140, colors=['#3498db', '#e74c3c'])
# # # #     plt.title('Proportion of Stress vs Non-Stress Predictions')
# # # #     st.pyplot(plt)

# # # # # Histogram of Message Lengths
# # # # st.write("### Histogram of Message Lengths")
# # # # message_lengths = chat_history_df['User'].apply(len)
# # # # plt.figure(figsize=(6, 4))
# # # # sns.histplot(message_lengths, bins=30, kde=True, color='purple')
# # # # plt.xlabel('Message Length')
# # # # plt.ylabel('Frequency')
# # # # plt.title('Distribution of Message Lengths')
# # # # st.pyplot(plt)

# # # # # Confusion Matrix Heatmap
# # # # st.write("### Confusion Matrix")
# # # # conf_matrix = confusion_matrix(y_test, y_pred)
# # # # plt.figure(figsize=(6, 4))
# # # # sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Stress', 'Stress'], yticklabels=['No Stress', 'Stress'])
# # # # plt.xlabel('Predicted')
# # # # plt.ylabel('Actual')
# # # # plt.title('Confusion Matrix')
# # # # st.pyplot(plt)

# # # # # Add some information about the app
# # # # st.write("""
# # # # #### How it works:
# # # # - This chatbot uses a Naive Bayes classifier trained on textual data to detect stress.
# # # # - Type any message above and click on 'Send' to see if stress is detected.
# # # # - The model preprocesses the text by cleaning it and converting it into numerical features using TF-IDF vectorization.
# # # # """)

# # # # # Display performance metrics
# # # # st.write("### Model Performance Metrics")
# # # # st.write(f"Accuracy: {accuracy:.2f}")
# # # # st.write(f"Precision: {precision:.2f}")
# # # # st.write(f"Recall: {recall:.2f}")
# # # # st.write(f"F1 Score: {f1:.2f}")

# # # # # Footer
# # # # st.markdown("""
# # # #     <div style='text-align: center; padding: 1rem; font-size: 0.9rem; color: #7f8c8d;'>
# # # #     Developed by Hassan Arzoo - Roll No: CSE204008, Guided by Dr. Abhishek Das, Associate Professor, CSE Department, Aliah University
# # # #     </div>
# # # # """, unsafe_allow_html=True)

# # # import streamlit as st
# # # import pandas as pd
# # # import numpy as np
# # # from sklearn.feature_extraction.text import TfidfVectorizer
# # # from sklearn.naive_bayes import MultinomialNB
# # # from sklearn.model_selection import train_test_split
# # # from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# # # import nltk
# # # import re
# # # from nltk.corpus import stopwords
# # # import string
# # # from wordcloud import WordCloud
# # # import matplotlib.pyplot as plt
# # # import seaborn as sns
# # # import os
# # # import time

# # # # Download stopwords
# # # nltk.download('stopwords')

# # # # Load data
# # # df = pd.read_csv('stress.csv')

# # # # Preprocess data
# # # stemmer = nltk.SnowballStemmer("english")
# # # stopword = set(stopwords.words('english'))

# # # def clean(text):
# # #     text = str(text).lower()
# # #     text = re.sub(r'\[.*?\]', '', text)
# # #     text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
# # #     text = re.sub(r'\w*\d\w*', '', text)
# # #     text = re.sub(r'[''""…]', '', text)
# # #     text = re.sub(r'\n', '', text)
# # #     text = [word for word in text.split(' ') if word not in stopword]
# # #     text = ' '.join(text)
# # #     text = [stemmer.stem(word) for word in text.split(' ')]
# # #     text = ' '.join(text)
# # #     return text

# # # df['text'] = df['text'].apply(clean)

# # # # Feature extraction
# # # tfidf = TfidfVectorizer(max_features=5000)
# # # X = tfidf.fit_transform(df['text']).toarray()
# # # y = df['label']

# # # # Train model
# # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # # model = MultinomialNB()
# # # model.fit(X_train, y_train)
# # # y_pred = model.predict(X_test)

# # # # Performance metrics
# # # accuracy = accuracy_score(y_test, y_pred)
# # # precision = precision_score(y_test, y_pred)
# # # recall = recall_score(y_test, y_pred)
# # # f1 = f1_score(y_test, y_pred)

# # # # Initialize chat history file
# # # CHAT_HISTORY_FILE = 'chat_history.csv'
# # # if not os.path.exists(CHAT_HISTORY_FILE):
# # #     pd.DataFrame(columns=['User', 'Bot']).to_csv(CHAT_HISTORY_FILE, index=False)

# # # # Custom CSS for styling
# # # st.markdown("""
# # #     <style>
# # #     .message-container {
# # #         margin-bottom: 1rem;
# # #     }
# # #     .message {
# # #         background-color: #ffffff;
# # #         padding: 1rem;
# # #         border-radius: 10px;
# # #         margin-bottom: 0.5rem;
# # #         font-family: cursive; /* Change font family to cursive */
# # #     }
# # #     .bot {
# # #         text-align: left;
# # #         color: #2c3e50;
# # #     }
# # #     .user {
# # #         text-align: right;
# # #         color: #2980b9;
# # #     }
# # #     .bold {
# # #         font-weight: bold;
# # #     }
# # #     </style>
# # # """, unsafe_allow_html=True)

# # # # Streamlit app layout
# # # st.write("""
# # # ### Stress Detection Chatbot
# # # Hi there! I'm here to help you detect stress. Go ahead and type your message below.
# # # """)

# # # # Chat interaction
# # # user_input = st.text_input("You:", "")
# # # if st.button("Send"):
# # #     if user_input:
# # #         with st.spinner('Calculating...'):
# # #             time.sleep(1)  # Simulate a delay
# # #             cleaned_input = clean(user_input)
# # #             vectorized_input = tfidf.transform([cleaned_input]).toarray()
# # #             prediction = model.predict(vectorized_input)
# # #             result = "<span class='bold'>Stress Detected</span>" if prediction[0] == 1 else "<span class='bold'>No Stress Detected</span>"

# # #             # Update chat history file
# # #             chat_history_df = pd.read_csv(CHAT_HISTORY_FILE)
# # #             chat_history_df = pd.concat([chat_history_df, pd.DataFrame({'User': [user_input], 'Bot': [result]})], ignore_index=True)
# # #             chat_history_df.to_csv(CHAT_HISTORY_FILE, index=False)

# # #             # Display chat messages
# # #             st.markdown(f"<div class='message user'>{user_input}</div>", unsafe_allow_html=True)
# # #             st.markdown(f"<div class='message bot'>{result}</div>", unsafe_allow_html=True)

# # # # Display chat history
# # # st.sidebar.title("Chat History")
# # # chat_history_df = pd.read_csv(CHAT_HISTORY_FILE)
# # # for index, row in chat_history_df.iterrows():
# # #     st.sidebar.markdown('<div class="message-container">', unsafe_allow_html=True)
# # #     st.sidebar.markdown('<div class="message user">' + row['User'] + '</div>', unsafe_allow_html=True)
# # #     st.sidebar.markdown('<div class="message bot">' + row['Bot'] + '</div>', unsafe_allow_html=True)
# # #     st.sidebar.markdown('</div>', unsafe_allow_html=True)

# # # # Display chat history graph
# # # st.sidebar.title("Chat History Graph")
# # # if not chat_history_df.empty:
# # #     chat_history_df['Bot'] = chat_history_df['Bot'].apply(lambda x: 1 if x == "<span class='bold'>Stress Detected</span>" else 0)
# # #     plt.figure(figsize=(6, 4))
# # #     sns.lineplot(data=chat_history_df, x=chat_history_df.index, y='Bot', markers=True)
# # #     plt.xlabel('Interaction')
# # #     plt.ylabel('Stress (1) / No Stress (0)')
# # #     plt.title('Chat History: Stress Detection Over Time')
# # #     st.sidebar.pyplot(plt)

# # # # Add some information about the app
# # # st.write("""
# # # #### How it works:
# # # - This chatbot uses a Naive Bayes classifier trained on textual data to detect stress.
# # # - Type any message above and click on 'Send' to see if stress is detected.
# # # - The model preprocesses the text by cleaning it and converting it into numerical features using TF-IDF vectorization.
# # # """)

# # # # Display performance metrics
# # # st.write("### Model Performance Metrics")
# # # st.write(f"Accuracy: {accuracy:.2f}")
# # # st.write(f"Precision: {precision:.2f}")
# # # st.write(f"Recall: {recall:.2f}")
# # # st.write(f"F1 Score: {f1:.2f}")

# # # # Footer
# # # st.markdown("""
# # #     <div style='text-align: center; padding: 1rem; font-size: 0.9rem; color: #7f8c8d;'>
# # #     Developed by Hassan Arzoo - Roll No: CSE204008, Guided by Dr. Abhishek Das, Associate Professor, CSE Department, Aliah University
# # #     </div>
# # # """, unsafe_allow_html=True)
# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # from sklearn.naive_bayes import MultinomialNB
# # from sklearn.model_selection import train_test_split
# # from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# # import nltk
# # import re
# # from nltk.corpus import stopwords
# # import string
# # from wordcloud import WordCloud
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # import os
# # import time

# # # Download stopwords
# # nltk.download('stopwords')

# # # Load data
# # df = pd.read_csv('stress.csv')

# # # Preprocess data
# # stemmer = nltk.SnowballStemmer("english")
# # stopword = set(stopwords.words('english'))

# # def clean(text):
# #     text = str(text).lower()
# #     text = re.sub(r'\[.*?\]', '', text)
# #     text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
# #     text = re.sub(r'\w*\d\w*', '', text)
# #     text = re.sub(r'[''""…]', '', text)
# #     text = re.sub(r'\n', '', text)
# #     text = [word for word in text.split(' ') if word not in stopword]
# #     text = ' '.join(text)
# #     text = [stemmer.stem(word) for word in text.split(' ')]
# #     text = ' '.join(text)
# #     return text

# # df['text'] = df['text'].apply(clean)

# # # Feature extraction
# # tfidf = TfidfVectorizer(max_features=5000)
# # X = tfidf.fit_transform(df['text']).toarray()
# # y = df['label']

# # # Train model
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # model = MultinomialNB()
# # model.fit(X_train, y_train)
# # y_pred = model.predict(X_test)

# # # Performance metrics
# # accuracy = accuracy_score(y_test, y_pred)
# # precision = precision_score(y_test, y_pred)
# # recall = recall_score(y_test, y_pred)
# # f1 = f1_score(y_test, y_pred)

# # # Initialize chat history file
# # CHAT_HISTORY_FILE = 'chat_history.csv'
# # if not os.path.exists(CHAT_HISTORY_FILE):
# #     pd.DataFrame(columns=['User', 'Bot']).to_csv(CHAT_HISTORY_FILE, index=False)

# # # Custom CSS for styling
# # st.markdown("""
# #     <style>
# #     .message-container {
# #         margin-bottom: 1rem;
# #     }
# #     .message {
# #         background-color: #ffffff;
# #         padding: 1rem;
# #         border-radius: 10px;
# #         margin-bottom: 0.5rem;
# #         font-family: cursive; /* Change font family to cursive */
# #     }
# #     .bot {
# #         text-align: left;
# #         color: #2c3e50;
# #     }
# #     .user {
# #         text-align: right;
# #         color: #2980b9;
# #     }
# #     .bold {
# #         font-weight: bold;
# #     }
# #     </style>
# # """, unsafe_allow_html=True)

# # # Streamlit app layout
# # st.write("""
# # ### Stress Detection Chatbot
# # Hi there! I'm here to help you detect stress. Go ahead and type your message below.
# # """)

# # # Chat interaction
# # user_input = st.text_input("You:", "")
# # if st.button("Send"):
# #     if user_input:
# #         with st.spinner('Calculating...'):
# #             time.sleep(1)  # Simulate a delay
# #             cleaned_input = clean(user_input)
# #             vectorized_input = tfidf.transform([cleaned_input]).toarray()
# #             prediction = model.predict(vectorized_input)
# #             result = "<span class='bold'>Stress Detected</span>" if prediction[0] == 1 else "<span class='bold'>No Stress Detected</span>"

# #             # Update chat history file
# #             chat_history_df = pd.read_csv(CHAT_HISTORY_FILE)
# #             chat_history_df = pd.concat([chat_history_df, pd.DataFrame({'User': [user_input], 'Bot': [result]})], ignore_index=True)
# #             chat_history_df.to_csv(CHAT_HISTORY_FILE, index=False)

# #             # Display chat messages
# #             st.markdown(f"<div class='message user'>{user_input}</div>", unsafe_allow_html=True)
# #             st.markdown(f"<div class='message bot'>{result}</div>", unsafe_allow_html=True)

# # # Display chat history
# # st.sidebar.title("Chat History")
# # chat_history_df = pd.read_csv(CHAT_HISTORY_FILE)
# # for index, row in chat_history_df.iterrows():
# #     st.sidebar.markdown('<div class="message-container">', unsafe_allow_html=True)
# #     st.sidebar.markdown('<div class="message user">' + row['User'] + '</div>', unsafe_allow_html=True)
# #     st.sidebar.markdown('<div class="message bot">' + row['Bot'] + '</div>', unsafe_allow_html=True)
# #     st.sidebar.markdown('</div>', unsafe_allow_html=True)

# # # Display chat history graph
# # st.sidebar.title("Chat History Graph")
# # if not chat_history_df.empty:
# #     chat_history_df['Bot'] = chat_history_df['Bot'].apply(lambda x: 1 if x == "<span class='bold'>Stress Detected</span>" else 0)
# #     plt.figure(figsize=(6, 4))
# #     sns.lineplot(data=chat_history_df, x=chat_history_df.index, y='Bot', markers=True)
# #     plt.xlabel('Interaction')
# #     plt.ylabel('Stress (1) / No Stress (0)')
# #     plt.title('Chat History: Stress Detection Over Time')
# #     st.sidebar.pyplot(plt)

# # # Footer
# # st.markdown("""
# #     <div style='text-align: center; padding: 1rem; font-size: 0.9rem; color: #7f8c8d;'>
# #     Developed by Hassan Arzoo (Roll No: CSE204008), guided by Dr. Abhishek Das, Associate Professor, CSE Dept., Aliah University.
# #     </div>
# # """, unsafe_allow_html=True)
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import nltk
# import re
# from nltk.corpus import stopwords
# import string
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# # Download stopwords
# nltk.download('stopwords')

# # Load data
# df = pd.read_csv('stress.csv')

# # Preprocess data
# stemmer = nltk.SnowballStemmer("english")
# stopword = set(stopwords.words('english'))

# def clean(text):
#     text = str(text).lower()
#     text = re.sub(r'\[.*?\]', '', text)
#     text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
#     text = re.sub(r'\w*\d\w*', '', text)
#     text = re.sub(r'[''""…]', '', text)
#     text = re.sub(r'\n', '', text)
#     text = [word for word in text.split(' ') if word not in stopword]
#     text = ' '.join(text)
#     text = [stemmer.stem(word) for word in text.split(' ')]
#     text = ' '.join(text)
#     return text

# df['text'] = df['text'].apply(clean)

# # Feature extraction
# tfidf = TfidfVectorizer(max_features=5000)
# X = tfidf.fit_transform(df['text']).toarray()
# y = df['label']

# # Train model
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = MultinomialNB()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# # Performance metrics
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)

# # Initialize chat history file
# CHAT_HISTORY_FILE = 'chat_history.csv'
# if not os.path.exists(CHAT_HISTORY_FILE):
#     pd.DataFrame(columns=['User', 'Bot']).to_csv(CHAT_HISTORY_FILE, index=False)

# # Custom CSS for styling
# st.markdown("""
#     <style>
#     .message-container {
#         margin-bottom: 1rem;
#     }
#     .message {
#         background-color: #ffffff;
#         padding: 1rem;
#         border-radius: 10px;
#         margin-bottom: 0.5rem;
#         font-family: cursive; /* Change font family to cursive */
#     }
#     .bot {
#         text-align: left;
#         color: #2c3e50;
#     }
#     .user {
#         text-align: right;
#         color: #2980b9;
#     }
#     .button {
#         background-color: #3498db;
#         color: white;
#         border: none;
#         padding: 1rem 2rem;
#         font-size: 1.2rem;
#         border-radius: 5px;
#         cursor: pointer;
#     }
#     .button:hover {
#         background-color: #2980b9;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # Streamlit app layout
# st.write("""
# ### Stress Detection Chatbot
# Hi there! I'm here to help you detect stress. Go ahead and type your message below.
# """)

# # # Chat interaction
# # user_input = st.text_input("You:", "")
# # if st.button("Send"):
# #     if user_input:
# #         # Clean user input
# #         cleaned_input = clean(user_input)
        
# #         # Vectorize input
# #         vectorized_input = tfidf.transform([cleaned_input]).toarray()
        
# #         # Predict
# #         prediction = model.predict(vectorized_input)
# #         result = "Stress Detected" if prediction[0] == 1 else "No Stress Detected"
        
# #         # Display bot's response
# #         st.write("Bot:", result)
        
# #         # Update chat history file
# #         chat_history_df = pd.read_csv(CHAT_HISTORY_FILE)
# #         new_entry = pd.DataFrame({'User': [user_input], 'Bot': [result]})
# #         chat_history_df = pd.concat([chat_history_df, new_entry], ignore_index=True)
# #         chat_history_df.to_csv(CHAT_HISTORY_FILE, index=False)
# # Chat interaction
# # user_input = st.text_input("You:", "")
# # if st.button("Send"):
# #     if user_input:
# #         # Clean user input
# #         cleaned_input = clean(user_input)
        
# #         # Vectorize input
# #         vectorized_input = tfidf.transform([cleaned_input]).toarray()
        
# #         # Predict
# #         prediction = model.predict(vectorized_input)
# #         result = "Stress Detected" if prediction[0] == 1 else "No Stress Detected"
        
# #         # Display bot's response in a dialogue box
# #         st.write(f"Bot: {result}")
        
# #         # Update chat history file
# #         chat_history_df = pd.read_csv(CHAT_HISTORY_FILE)
# #         new_entry = pd.DataFrame({'User': [user_input], 'Bot': [result]})
# #         chat_history_df = pd.concat([chat_history_df, new_entry], ignore_index=True)
# #         chat_history_df.to_csv(CHAT_HISTORY_FILE, index=False)
# # Chat interaction
# user_input = st.text_input("You:", "")
# placeholder = st.empty()  # Placeholder for bot's reply

# if st.button("Send"):
#     if user_input:
#         # Clean user input
#         cleaned_input = clean(user_input)
        
#         # Vectorize input
#         vectorized_input = tfidf.transform([cleaned_input]).toarray()
        
#         # Predict
#         prediction = model.predict(vectorized_input)
#         result = "Stress Detected" if prediction[0] == 1 else "No Stress Detected"
        
#         # Display bot's response in a dialogue box
#         placeholder.write(f"Bot: {result}")
        
#         # Update chat history file
#         chat_history_df = pd.read_csv(CHAT_HISTORY_FILE)
#         new_entry = pd.DataFrame({'User': [user_input], 'Bot': [result]})
#         chat_history_df = pd.concat([chat_history_df, new_entry], ignore_index=True)
#         chat_history_df.to_csv(CHAT_HISTORY_FILE, index=False)

# # Display chat history
# st.sidebar.title("Chat History")
# chat_history_df = pd.read_csv(CHAT_HISTORY_FILE)
# for index, row in chat_history_df.iterrows():
#     st.sidebar.markdown('<div class="message-container">', unsafe_allow_html=True)
#     st.sidebar.markdown('<div class="message user">' + row['User'] + '</div>', unsafe_allow_html=True)
#     st.sidebar.markdown('<div class="message bot">' + row['Bot'] + '</div>', unsafe_allow_html=True)
#     st.sidebar.markdown('</div>', unsafe_allow_html=True)

# # Display word cloud
# st.write("### Word Cloud of the Text Data")
# wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['text']))
# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# st.pyplot(plt)

# # Add some information about the app
# st.write("""
# #### How it works:
# - This chatbot uses a Naive Bayes classifier trained on textual data to detect stress.
# - Type any message above and click on 'Send' to see if stress is detected.
# - The model preprocesses the text by cleaning it and converting it into numerical features using TF-IDF vectorization.
# """)

# # Display performance metrics
# st.write("### Model Performance Metrics")
# st.write(f"Accuracy: {accuracy:.2f}")
# st.write(f"Precision: {precision:.2f}")
# st.write(f"Recall: {recall:.2f}")
# st.write(f"F1 Score: {f1:.2f}")

# # Distribution of Stress and Non-Stress Labels
# st.write("### Distribution of Stress and Non-Stress Labels")
# label_counts = df['label'].value_counts()
# plt.figure(figsize=(6, 4))
# sns.barplot(x=label_counts.index, y=label_counts.values, palette='viridis')
# plt.xlabel('Label')
# plt.ylabel('Count')
# plt.title('Distribution of Stress and Non-Stress Labels')
# st.pyplot(plt)

# # Most common words in stress-related and non-stress-related texts
# st.write("### Most Common Words in Stress-Related Texts")
# stress_texts = ' '.join(df[df['label'] == 1]['text'])
# stress_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(stress_texts)
# plt.figure(figsize=(10, 5))
# plt.imshow(stress_wordcloud, interpolation='bilinear')
# plt.axis('off')
# st.pyplot(plt)

# st.write("### Most Common Words in Non-Stress-Related Texts")
# non_stress_texts = ' '.join(df[df['label'] == 0]['text'])
# non_stress_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(non_stress_texts)
# plt.figure(figsize=(10, 5))
# plt.imshow(non_stress_wordcloud, interpolation='bilinear')
# plt.axis('off')
# st.pyplot(plt)

# # Footer
# st.markdown("""
#     <div style='text-align: center; padding: 1rem; font-size: 0.9rem; color: #7f8c8d;'>
#     Developed by Hassan Arzoo - Roll No: CSE204008, Guided by Dr. Abhishek Das, Associate Professor, CSE Department, Aliah University
#     </div>
# """, unsafe_allow_html=True)
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import nltk
# import re
# from nltk.corpus import stopwords
# import string
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# # Download stopwords
# nltk.download('stopwords')

# # Load data
# df = pd.read_csv('stress.csv')

# # Preprocess data
# stemmer = nltk.SnowballStemmer("english")
# stopword = set(stopwords.words('english'))

# def clean(text):
#     text = str(text).lower()
#     text = re.sub(r'\[.*?\]', '', text)
#     text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
#     text = re.sub(r'\w*\d\w*', '', text)
#     text = re.sub(r'[''""…]', '', text)
#     text = re.sub(r'\n', '', text)
#     text = [word for word in text.split(' ') if word not in stopword]
#     text = ' '.join(text)
#     text = [stemmer.stem(word) for word in text.split(' ')]
#     text = ' '.join(text)
#     return text

# df['text'] = df['text'].apply(clean)

# # Feature extraction
# tfidf = TfidfVectorizer(max_features=5000)
# X = tfidf.fit_transform(df['text']).toarray()
# y = df['label']

# # Train model
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = MultinomialNB()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# # Performance metrics
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)

# # Initialize chat history file
# CHAT_HISTORY_FILE = 'chat_history.csv'
# if not os.path.exists(CHAT_HISTORY_FILE):
#     pd.DataFrame(columns=['User', 'Bot']).to_csv(CHAT_HISTORY_FILE, index=False)

# # Custom CSS for styling
# st.markdown("""
#     <style>
#     .message-container {
#         margin-bottom: 1rem;
#     }
#     .message {
#         background-color: #ffffff;
#         padding: 1rem;
#         border-radius: 10px;
#         margin-bottom: 0.5rem;
#         font-family: cursive; /* Change font family to cursive */
#     }
#     .bot {
#         text-align: left;
#         color: #2c3e50;
#     }
#     .user {
#         text-align: right;
#         color: #2980b9;
#     }
#     .button {
#         background-color: #3498db;
#         color: white;
#         border: none;
#         padding: 1rem 2rem;
#         font-size: 1.2rem;
#         border-radius: 5px;
#         cursor: pointer;
#     }
#     .button:hover {
#         background-color: #2980b9;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # Streamlit app layout
# st.write("""
# ### Stress Detection Chatbot
# Hi there! I'm here to help you detect stress. Go ahead and type your message below.
# """)

# # Chat interaction
# user_input = st.text_input("You:", "")
# if st.button("Send"):
#     if user_input:
#         cleaned_input = clean(user_input)
#         vectorized_input = tfidf.transform([cleaned_input]).toarray()
#         prediction = model.predict(vectorized_input)
#         result = "Stress Detected" if prediction[0] == 1 else "No Stress Detected"

#         # Update chat history file
#         chat_history_df = pd.read_csv(CHAT_HISTORY_FILE)
#         chat_history_df = pd.concat([chat_history_df, pd.DataFrame({'User': [user_input], 'Bot': [result]})], ignore_index=True)
#         chat_history_df.to_csv(CHAT_HISTORY_FILE, index=False)

# # Display chat history
# st.sidebar.title("Chat History")
# chat_history_df = pd.read_csv(CHAT_HISTORY_FILE)
# for index, row in chat_history_df.iterrows():
#     st.sidebar.markdown('<div class="message-container">', unsafe_allow_html=True)
#     st.sidebar.markdown('<div class="message user">' + row['User'] + '</div>', unsafe_allow_html=True)
#     st.sidebar.markdown('<div class="message bot">' + row['Bot'] + '</div>', unsafe_allow_html=True)
#     st.sidebar.markdown('</div>', unsafe_allow_html=True)

# # Display chat history graph
# st.sidebar.title("Chat History Graph")
# if not chat_history_df.empty:
#     chat_history_df['Bot'] = chat_history_df['Bot'].apply(lambda x: 1 if x == "Stress Detected" else 0)
#     plt.figure(figsize=(6, 4))
#     sns.lineplot(data=chat_history_df, x=chat_history_df.index, y='Bot', markers=True)
#     plt.xlabel('Interaction')
#     plt.ylabel('Stress (1) / No Stress (0)')
#     plt.title('Chat History: Stress Detection Over Time')
#     st.sidebar.pyplot(plt)

# # Display word cloud
# st.write("### Word Cloud of the Text Data")
# wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['text']))
# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# st.pyplot(plt)

# # Add some information about the app
# st.write("""
# #### How it works:
# - This chatbot uses a Naive Bayes classifier trained on textual data to detect stress.
# - Type any message above and click on 'Send' to see if stress is detected.
# - The model preprocesses the text by cleaning it and converting it into numerical features using TF-IDF vectorization.
# """)

# # Display performance metrics
# st.write("### Model Performance Metrics")
# st.write(f"Accuracy: {accuracy:.2f}")
# st.write(f"Precision: {precision:.2f}")
# st.write(f"Recall: {recall:.2f}")
# st.write(f"F1 Score: {f1:.2f}")

# # Distribution of Stress and Non-Stress Labels
# st.write("### Distribution of Stress and Non-Stress Labels")
# label_counts = df['label'].value_counts()
# plt.figure(figsize=(6, 4))
# sns.barplot(x=label_counts.index, y=label_counts.values, palette='viridis')
# plt.xlabel('Label')
# plt.ylabel('Count')
# plt.title('Distribution of Stress and Non-Stress Labels')
# st.pyplot(plt)

# # Most common words in stress-related and non-stress-related texts
# st.write("### Most Common Words in Stress-Related Texts")
# stress_texts = ' '.join(df[df['label'] == 1]['text'])
# stress_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(stress_texts)
# plt.figure(figsize=(10, 5))
# plt.imshow(stress_wordcloud, interpolation
# ='bilinear')
# plt.axis('off')
# st.pyplot(plt)

# st.write("### Most Common Words in Non-Stress-Related Texts")
# non_stress_texts = ' '.join(df[df['label'] == 0]['text'])
# non_stress_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(non_stress_texts)
# plt.figure(figsize=(10, 5))
# plt.imshow(non_stress_wordcloud, interpolation='bilinear')
# plt.axis('off')
# st.pyplot(plt)

# # Footer
# st.markdown("""
#     <div style='text-align: center; padding: 1rem; font-size: 0.9rem; color: #7f8c8d;'>
#     Developed by Hassan Arzoo - Roll No: CSE204008, Guided by Dr. Abhishek Das, Associate Professor, CSE Department, Aliah University
#     </div>
# """, unsafe_allow_html=True)
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import nltk
# import re
# from nltk.corpus import stopwords
# import string
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# # Download stopwords
# nltk.download('stopwords')

# # Load data
# df = pd.read_csv('stress.csv')

# # Preprocess data
# stemmer = nltk.SnowballStemmer("english")
# stopword = set(stopwords.words('english'))

# def clean(text):
#     text = str(text).lower()
#     text = re.sub(r'\[.*?\]', '', text)
#     text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
#     text = re.sub(r'\w*\d\w*', '', text)
#     text = re.sub(r'[''""…]', '', text)
#     text = re.sub(r'\n', '', text)
#     text = [word for word in text.split(' ') if word not in stopword]
#     text = ' '.join(text)
#     text = [stemmer.stem(word) for word in text.split(' ')]
#     text = ' '.join(text)
#     return text

# df['text'] = df['text'].apply(clean)

# # Feature extraction
# tfidf = TfidfVectorizer(max_features=5000)
# X = tfidf.fit_transform(df['text']).toarray()
# y = df['label']

# # Train model
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = MultinomialNB()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# # Performance metrics
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)

# # Initialize chat history file
# CHAT_HISTORY_FILE = 'chat_history.csv'
# if not os.path.exists(CHAT_HISTORY_FILE):
#     pd.DataFrame(columns=['User', 'Bot']).to_csv(CHAT_HISTORY_FILE, index=False)

# # Custom CSS for styling
# st.markdown("""
#     <style>
#     .message-container {
#         margin-bottom: 1rem;
#     }
#     .message {
#         background-color: #ffffff;
#         padding: 1rem;
#         border-radius: 10px;
#         margin-bottom: 0.5rem;
#         font-family: cursive; /* Change font family to cursive */
#     }
#     .bot {
#         text-align: left;
#         color: #2c3e50;
#     }
#     .user {
#         text-align: right;
#         color: #2980b9;
#     }
#     .button {
#         background-color: #3498db;
#         color: white;
#         border: none;
#         padding: 1rem 2rem;
#         font-size: 1.2rem;
#         border-radius: 5px;
#         cursor: pointer;
#     }
#     .button:hover {
#         background-color: #2980b9;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # Streamlit app layout
# st.write("""
# ### Stress Detection Chatbot
# Hi there! I'm here to help you detect stress. Go ahead and type your message below.
# """)

# # Chat interaction
# user_input = st.text_input("You:", "")
# if st.button("Send"):
#     if user_input:
#         cleaned_input = clean(user_input)
#         vectorized_input = tfidf.transform([cleaned_input]).toarray()
#         prediction = model.predict(vectorized_input)
#         result = "Stress Detected" if prediction[0] == 1 else "No Stress Detected"

#         # Display chatbot's response in the dialogue box
#         st.text_area("Bot:", value=result, height=100, max_chars=None, key=None)

#         # Update chat history file
#         chat_history_df = pd.read_csv(CHAT_HISTORY_FILE)
#         chat_history_df = pd.concat([chat_history_df, pd.DataFrame({'User': [user_input], 'Bot': [result]})], ignore_index=True)
#         chat_history_df.to_csv(CHAT_HISTORY_FILE, index=False)

# # Display word cloud
# st.write("### Word Cloud of the Text Data")
# wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['text']))
# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# st.pyplot(plt)

# # Add some information about the app
# st.write("""
# #### How it works:
# - This chatbot uses a Naive Bayes classifier trained on textual data to detect stress.
# - Type any message above and click on 'Send' to see if stress is detected.
# - The model preprocesses the text by cleaning it and converting it into numerical features using TF-IDF vectorization.
# """)

# # Display performance metrics
# st.write("### Model Performance Metrics")
# st.write(f"Accuracy: {accuracy:.2f}")
# st.write(f"Precision: {precision:.2f}")
# st.write(f"Recall: {recall:.2f}")
# st.write(f"F1 Score: {f1:.2f}")

# # Distribution of Stress and Non-Stress Labels
# st.write("### Distribution of Stress and Non-Stress Labels")
# label_counts = df['label'].value_counts()
# plt.figure(figsize=(6, 4))
# sns.barplot(x=label_counts.index, y=label_counts.values, palette='viridis')
# plt.xlabel('Label')
# plt.ylabel('Count')
# plt.title('Distribution of Stress and Non-Stress Labels')
# st.pyplot(plt)

# # Most common words in stress-related and non-stress-related texts
# st.write("### Most Common Words in Stress-Related Texts")
# stress_texts = ' '.join(df[df['label'] == 1]['text'])
# stress_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(stress_texts)
# plt.figure(figsize=(10, 5))
# plt.imshow(stress_wordcloud, interpolation='bilinear')
# plt.axis('off')
# st.pyplot(plt)

# st.write("### Most Common Words in Non-Stress-Related Texts")
# non_stress_texts = ' '.join(df[df['label'] == 0]['text'])
# non_stress_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(non_stress_texts)
# plt.figure(figsize=(10, 5))
# plt.imshow(non_stress_wordcloud, interpolation='bilinear')
# plt.axis('off')
# st.pyplot(plt)

# # Display chat history
# st.sidebar.title("Chat History")
# chat_history_df = pd.read_csv(CHAT_HISTORY_FILE)
# for index, row in chat_history_df.iterrows():
#     st.sidebar.markdown('<div class="message-container">', unsafe_allow_html=True)
#     st.sidebar.markdown('<div class="message user">' + row['User'] + '</div>', unsafe_allow_html=True)
#     st.sidebar.markdown('<div class="message bot">' + row['Bot'] + '</div>', unsafe_allow_html=True)
#     st.sidebar.markdown('</div>', unsafe_allow_html=True)

# # Display chat history graph
# st.sidebar.title("Chat History Graph")
# if not chat_history_df.empty:
#     chat_history_df['Bot'] = chat_history_df['Bot'].apply(lambda x: 1 if x == "Stress Detected" else 0)
#     plt.figure(figsize=(6, 4))
#     sns.lineplot(data=chat_history_df, x=chat_history_df.index, y='Bot', markers=True)
#     plt.xlabel('Interaction')
#     plt.ylabel('Stress (1) / No Stress (0)')
#     plt.title('Chat History: Stress Detection Over Time')
#     st.sidebar.pyplot(plt)

# # Footer
# st.markdown("""
#     <div style='text-align: center; padding: 1rem; font-size: 0.9rem; color: #7f8c8d;'>
#     Developed by Hassan Arzoo - Roll No: CSE204008, Guided by Dr. Abhishek Das, Associate Professor, CSE Department, Aliah University
#     </div>
# """, unsafe_allow_html=True)

# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# import nltk
# import re
# from nltk.corpus import stopwords
# import string
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px
# import joblib
# import os

# # Download stopwords
# nltk.download('stopwords')

# # Load data
# try:
#     df = pd.read_csv('stress.csv')
# except FileNotFoundError:
#     st.error("The stress.csv file was not found.")
#     st.stop()

# # Preprocess data
# stemmer = nltk.SnowballStemmer("english")
# stopword = set(stopwords.words('english'))

# def clean(text):
#     text = str(text).lower()
#     text = re.sub(r'\[.*?\]', '', text)
#     text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
#     text = re.sub(r'\w*\d\w*', '', text)
#     text = re.sub(r'[''""…]', '', text)
#     text = re.sub(r'\n', '', text)
#     text = [word for word in text.split() if word not in stopword]
#     text = ' '.join(text)
#     text = [stemmer.stem(word) for word in text.split()]
#     text = ' '.join(text)
#     return text

# df['text'] = df['text'].apply(clean)

# # Feature extraction
# tfidf = TfidfVectorizer(max_features=5000)
# X = tfidf.fit_transform(df['text']).toarray()
# y = df['label']

# # Train model
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = MultinomialNB()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# # Performance metrics
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# conf_matrix = confusion_matrix(y_test, y_pred)

# # Save model and vectorizer
# joblib.dump(model, 'naive_bayes_model.pkl')
# joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

# # Initialize chat history file
# CHAT_HISTORY_FILE = 'chat_history.csv'
# if not os.path.exists(CHAT_HISTORY_FILE):
#     pd.DataFrame(columns=['User', 'Bot']).to_csv(CHAT_HISTORY_FILE, index=False)

# # Custom CSS for styling
# st.markdown("""
#     <style>
#     .message-container {
#         margin-bottom: 1rem;
#     }
#     .message {
#         background-color: #ffffff;
#         padding: 1rem;
#         border-radius: 10px;
#         margin-bottom: 0.5rem;
#         font-family: cursive; /* Change font family to cursive */
#     }
#     .bot {
#         text-align: left;
#         color: #2c3e50;
#     }
#     .user {
#         text-align: right;
#         color: #2980b9;
#     }
#     .button {
#         background-color: #3498db;
#         color: white;
#         border: none;
#         padding: 1rem 2rem;
#         font-size: 1.2rem;
#         border-radius: 5px;
#         cursor: pointer;
#     }
#     .button:hover {
#         background-color: #2980b9;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # Streamlit app layout
# st.write("""
# ### Stress Detection Chatbot
# Hi there! I'm here to help you detect stress. Go ahead and type your message below.
# """)

# # Chat interaction
# user_input = st.text_area("You:", "", height=100)
# if st.button("Send"):
#     if user_input:
#         cleaned_input = clean(user_input)
#         vectorized_input = tfidf.transform([cleaned_input]).toarray()
#         prediction = model.predict(vectorized_input)
#         result = "Stress Detected" if prediction[0] == 1 else "No Stress Detected"

#         # Display chatbot's response in the dialogue box
#         st.text_area("Bot:", value=result, height=100, max_chars=None, key=None)

#         # Update chat history file
#         try:
#             chat_history_df = pd.read_csv(CHAT_HISTORY_FILE)
#             chat_history_df = pd.concat([chat_history_df, pd.DataFrame({'User': [user_input], 'Bot': [result]})], ignore_index=True)
#             chat_history_df.to_csv(CHAT_HISTORY_FILE, index=False)
#         except Exception as e:
#             st.error(f"An error occurred while updating the chat history: {e}")

# # Display word cloud
# st.write("### Word Cloud of the Text Data")
# wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['text']))
# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# st.pyplot(plt)

# # Add some information about the app
# st.write("""
# #### How it works:
# - This chatbot uses a Naive Bayes classifier trained on textual data to detect stress.
# - Type any message above and click on 'Send' to see if stress is detected.
# - The model preprocesses the text by cleaning it and converting it into numerical features using TF-IDF vectorization.
# """)

# # Display performance metrics
# st.write("### Model Performance Metrics")
# st.write(f"Accuracy: {accuracy:.2f}")
# st.write(f"Precision: {precision:.2f}")
# st.write(f"Recall: {recall:.2f}")
# st.write(f"F1 Score: {f1:.2f}")

# # Confusion matrix
# fig = plt.figure(figsize=(6, 4))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# st.pyplot(fig)

# # Distribution of Stress and Non-Stress Labels
# st.write("### Distribution of Stress and Non-Stress Labels")
# label_counts = df['label'].value_counts()
# fig = px.bar(label_counts, x=label_counts.index, y=label_counts.values, labels={'index': 'Label', 'y': 'Count'}, title='Distribution of Stress and Non-Stress Labels')
# st.plotly_chart(fig)

# # Most common words in stress-related and non-stress-related texts
# st.write("### Most Common Words in Stress-Related Texts")
# stress_texts = ' '.join(df[df['label'] == 1]['text'])
# stress_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(stress_texts)
# plt.figure(figsize=(10, 5))
# plt.imshow(stress_wordcloud, interpolation='bilinear')
# plt.axis('off')
# st.pyplot(plt)

# st.write("### Most Common Words in Non-Stress-Related Texts")
# non_stress_texts = ' '.join(df[df['label'] == 0]['text'])
# non_stress_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(non_stress_texts)
# plt.figure(figsize=(10, 5))
# plt.imshow(non_stress_wordcloud, interpolation='bilinear')
# plt.axis('off')
# st.pyplot(plt)

# # Display chat history
# st.sidebar.title("Chat History")
# try:
#     chat_history_df = pd.read_csv(CHAT_HISTORY_FILE)
#     for index, row in chat_history_df.iterrows():
#         st.sidebar.markdown('<div class="message-container">', unsafe_allow_html=True)
#         st.sidebar.markdown('<div class="message user">' + row['User'] + '</div>', unsafe_allow_html=True)
#         st.sidebar.markdown('<div class="message bot">' + row['Bot'] + '</div>', unsafe_allow_html=True)
#         st.sidebar.markdown('</div>', unsafe_allow_html=True)
# except Exception as e:
#     st.sidebar.error(f"An error occurred while loading the chat history: {e}")

# # Display chat history graph
# st.sidebar.title("Chat History Graph")
# if not chat_history_df.empty:
#     chat_history_df['Bot'] = chat_history_df['Bot'].apply(lambda x: 1 if x == "Stress Detected" else 0)
#     fig = px.line(chat_history_df, x=chat_history_df.index, y='Bot', markers=True, labels={'index': 'Interaction', 'Bot': 'Stress (1) / No Stress (0)'}, title='Chat History: Stress Detection Over Time')
#     st.sidebar.plotly_chart(fig)

# # Footer
# st.markdown("""
#     <div style='text-align: center; padding: 1rem; font-size: 0.9rem; color: #7f8c8d;'>
#     Developed by Hassan Arzoo - Roll No: CSE204008, Guided by Dr. Abhishek Das, Associate Professor, CSE Department, Aliah University
#     </div>
# """, unsafe_allow_html=True)


# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# import nltk
# import re
# from nltk.corpus import stopwords
# import string
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px
# import joblib
# import os

# # Download stopwords
# nltk.download('stopwords')

# # Load data
# try:
#     df = pd.read_csv('stress.csv')
# except FileNotFoundError:
#     st.error("The stress.csv file was not found.")
#     st.stop()

# # Preprocess data
# stemmer = nltk.SnowballStemmer("english")
# stopword = set(stopwords.words('english'))

# def clean(text):
#     text = str(text).lower()
#     text = re.sub(r'\[.*?\]', '', text)
#     text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
#     text = re.sub(r'\w*\d\w*', '', text)
#     text = re.sub(r'[''""…]', '', text)
#     text = re.sub(r'\n', '', text)
#     text = [word for word in text.split() if word not in stopword]
#     text = ' '.join(text)
#     text = [stemmer.stem(word) for word in text.split()]
#     text = ' '.join(text)
#     return text

# df['text'] = df['text'].apply(clean)

# # Feature extraction
# tfidf = TfidfVectorizer(max_features=5000)
# X = tfidf.fit_transform(df['text']).toarray()
# y = df['label']

# # Train model
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = MultinomialNB()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# # Performance metrics
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# conf_matrix = confusion_matrix(y_test, y_pred)

# # Save model and vectorizer
# joblib.dump(model, 'naive_bayes_model.pkl')
# joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

# # Initialize chat history file
# CHAT_HISTORY_FILE = 'chat_history.csv'
# if not os.path.exists(CHAT_HISTORY_FILE):
#     pd.DataFrame(columns=['User', 'Bot']).to_csv(CHAT_HISTORY_FILE, index=False)

# # Custom CSS for styling
# st.markdown("""
#     <style>
#     .message-container {
#         margin-bottom: 1rem;
#     }
#     .message {
#         background-color: #ffffff;
#         padding: 1rem;
#         border-radius: 10px;
#         margin-bottom: 0.5rem;
#         font-family: cursive; /* Change font family to cursive */
#     }
#     .bot {
#         text-align: left;
#         color: #2c3e50;
#     }
#     .user {
#         text-align: right;
#         color: #2980b9;
#     }
#     .button {
#         background-color: #3498db;
#         color: white;
#         border: none;
#         padding: 1rem 2rem;
#         font-size: 1.2rem;
#         border-radius: 5px;
#         cursor: pointer;
#     }
#     .button:hover {
#         background-color: #2980b9;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # Streamlit app layout
# st.write("""
# ### Stress Detection Chatbot
# Hi there! I'm here to help you detect stress. Go ahead and type your message below.
# """)

# # Initialize chat history
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []

# # Function to add a message to the chat history
# def add_message(user_msg, bot_msg):
#     st.session_state.chat_history.append((user_msg, bot_msg))

# # # 1Function to delete a specific message from the chat history
# # def delete_message(index):
# #     if 0 <= index < len(st.session_state.chat_history):
# #         st.session_state.chat_history.pop(index)
# # 2Function to delete a specific message from the chat history
# # def delete_message(index):
# #     if 0 <= index < len(st.session_state.chat_history):
# #         st.session_state.chat_history.pop(index)
# # 3Function to delete a specific message from the chat history
# # def delete_message(index):
# #     if 0 <= index < len(st.session_state.chat_history):
# #         del st.session_state.chat_history[index]
# # 4Function to delete a specific message from the chat history
# def delete_message(index):
#     if 0 <= index < len(st.session_state.chat_history):
#         del st.session_state.chat_history[index]
# # Function to clear all chat history
# def clear_history():
#     st.session_state.chat_history = []

# # Chat interaction
# user_input = st.text_area("You:", "", height=100)
# if st.button("Send"):
#     if user_input:
#         cleaned_input = clean(user_input)
#         vectorized_input = tfidf.transform([cleaned_input]).toarray()
#         prediction = model.predict(vectorized_input)
#         result = "Stress Detected" if prediction[0] == 1 else "No Stress Detected"

#         # Add to chat history
#         add_message(user_input, result)
        
#         # Display chatbot's response in the dialogue box
#         st.text_area("Bot:", value=result, height=100, max_chars=None, key=None)

#         # Update chat history file
#         try:
#             chat_history_df = pd.read_csv(CHAT_HISTORY_FILE)
#             chat_history_df = pd.concat([chat_history_df, pd.DataFrame({'User': [user_input], 'Bot': [result]})], ignore_index=True)
#             chat_history_df.to_csv(CHAT_HISTORY_FILE, index=False)
#         except Exception as e:
#             st.error(f"An error occurred while updating the chat history: {e}")

# # # 1Display chat history
# # st.write("### Chat History")
# # for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
# #     st.write(f"**You**: {user_msg}")
# #     st.write(f"**Bot**: {bot_msg}")
# #     if st.button(f"Delete message {i + 1}", key=f"delete_{i}"):
# #         delete_message(i)
# #         st.experimental_rerun()
# # 2Display chat history
# # st.write("### Chat History")
# # for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
# #     st.write(f"**You**: {user_msg}")
# #     st.write(f"**Bot**: {bot_msg}")
# #     delete_button_key = f"delete_{i}"  # Unique key for each delete button
# #     delete_button_clicked = st.button(f"Delete message {i + 1}", key=delete_button_key)
# #     if delete_button_clicked:
# #         delete_message(i)
# # 3Display chat history
# # st.write("### Chat History")
# # for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
# #     st.write(f"**You**: {user_msg}")
# #     st.write(f"**Bot**: {bot_msg}")
# #     delete_button_clicked = st.button(f"Delete message {i + 1}")
# #     if delete_button_clicked:
# #         delete_message(i)
# # 4Display chat history
# st.write("### Chat History")
# for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
#     delete_button_clicked = st.button(f"Delete message {i + 1}")
#     if delete_button_clicked:
#         delete_message(i)
#         st.experimental_rerun()  # Rerun the app to reflect the updated chat history
#     else:
#         st.write(f"**You**: {user_msg}")
#         st.write(f"**Bot**: {bot_msg}")
#         # Rerun the app to reflect the updated chat history
#         st.experimental_rerun()
#         # Update session state to remove the deleted message
#         st.session_state.chat_history = st.session_state.chat_history[:i] + st.session_state.chat_history[i+1:]

# # Button to clear all chat history
# if st.button("Clear all history"):
#     clear_history()
#     st.experimental_rerun()

# # Display word cloud
# st.write("### Word Cloud of the Text Data")
# wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['text']))
# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# st.pyplot(plt)

# # Add some information about the app
# st.write("""
# #### How it works:
# - This chatbot uses a Naive Bayes classifier trained on textual data to detect stress.
# - Type any message above and click on 'Send' to see if stress is detected.
# - The model preprocesses the text by cleaning it and converting it into numerical features using TF-IDF vectorization.
# """)

# # Display performance metrics
# st.write("### Model Performance Metrics")
# st.write(f"Accuracy: {accuracy:.2f}")
# st.write(f"Precision: {precision:.2f}")
# st.write(f"Recall: {recall:.2f}")
# st.write(f"F1 Score: {f1:.2f}")

# # Confusion matrix
# fig = plt.figure(figsize=(6, 4))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# st.pyplot(fig)

# # Distribution of Stress and Non-Stress Labels
# st.write("### Distribution of Stress and Non-Stress Labels")
# label_counts = df['label'].value_counts()
# fig = px.bar(label_counts, x=label_counts.index, y=label_counts.values, labels={'index': 'Label', 'y': 'Count'}, title='Distribution of Stress and Non-Stress Labels')
# st.plotly_chart(fig)

# # Most common words in stress-related and non-stress-related texts
# st.write("### Most Common Words in Stress-Related Texts")
# stress_texts = ' '.join(df[df['label'] == 1]['text'])
# stress_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(stress_texts)
# plt.figure(figsize=(10, 5))
# plt.imshow(stress_wordcloud, interpolation='bilinear')
# plt.axis('off')
# st.pyplot(plt)

# st.write("### Most Common Words in Non-Stress-Related Texts")
# non_stress_texts = ' '.join(df[df['label'] == 0]['text'])
# non_stress_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(non_stress_texts)
# plt.figure(figsize=(10, 5))
# plt.imshow(non_stress_wordcloud, interpolation='bilinear')
# plt.axis('off')
# st.pyplot(plt)

# # Display chat history in sidebar
# st.sidebar.title("Chat History")
# try:
#     chat_history_df = pd.read_csv(CHAT_HISTORY_FILE)
#     for index, row in chat_history_df.iterrows():
#         st.sidebar.markdown('<div class="message-container">', unsafe_allow_html=True)
#         st.sidebar.markdown('<div class="message user">' + row['User'] + '</div>', unsafe_allow_html=True)
#         st.sidebar.markdown('<div class="message bot">' + row['Bot'] + '</div>', unsafe_allow_html=True)
#         st.sidebar.markdown('</div>', unsafe_allow_html=True)
# except Exception as e:
#     st.sidebar.error(f"An error occurred while loading the chat history: {e}")

# # Display chat history graph in sidebar
# st.sidebar.title("Chat History Graph")
# if not chat_history_df.empty:
#     chat_history_df['Bot'] = chat_history_df['Bot'].apply(lambda x: 1 if x == "Stress Detected" else 0)
#     fig = px.line(chat_history_df, x=chat_history_df.index, y='Bot', markers=True, labels={'index': 'Interaction', 'Bot': 'Stress (1) / No Stress (0)'}, title='Chat History: Stress Detection Over Time')
#     st.sidebar.plotly_chart(fig)

# # Footer
# st.markdown("""
#     <div style='text-align: center; padding: 1rem; font-size: 0.9rem; color: #7f8c8d;'>
#     Developed by Hassan Arzoo - Roll No: CSE204008, Guided by Dr. Abhishek Das, Associate Professor, CSE Department, Aliah University
#     </div>
# """, unsafe_allow_html=True)


# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# import nltk
# import re
# from nltk.corpus import stopwords
# import string
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px
# import joblib
# import os

# # Download stopwords
# nltk.download('stopwords')

# # Load data
# try:
#     df = pd.read_csv('stress.csv')
# except FileNotFoundError:
#     st.error("The stress.csv file was not found.")
#     st.stop()

# # Preprocess data
# stemmer = nltk.SnowballStemmer("english")
# stopword = set(stopwords.words('english'))

# def clean(text):
#     text = str(text).lower()
#     text = re.sub(r'\[.*?\]', '', text)
#     text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
#     text = re.sub(r'\w*\d\w*', '', text)
#     text = re.sub(r'[''""…]', '', text)
#     text = re.sub(r'\n', '', text)
#     text = [word for word in text.split() if word not in stopword]
#     text = ' '.join(text)
#     text = [stemmer.stem(word) for word in text.split()]
#     text = ' '.join(text)
#     return text

# df['text'] = df['text'].apply(clean)

# # Feature extraction
# tfidf = TfidfVectorizer(max_features=5000)
# X = tfidf.fit_transform(df['text']).toarray()
# y = df['label']

# # Train model
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = MultinomialNB()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# # Performance metrics
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# conf_matrix = confusion_matrix(y_test, y_pred)

# # Save model and vectorizer
# joblib.dump(model, 'naive_bayes_model.pkl')
# joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

# # Initialize chat history file
# CHAT_HISTORY_FILE = 'chat_history.csv'
# if not os.path.exists(CHAT_HISTORY_FILE):
#     pd.DataFrame(columns=['User', 'Bot']).to_csv(CHAT_HISTORY_FILE, index=False)

# # Custom CSS for styling
# st.markdown("""
#     <style>
#     .message-container {
#         margin-bottom: 1rem;
#     }
#     .message {
#         background-color: #ffffff;
#         padding: 1rem;
#         border-radius: 10px;
#         margin-bottom: 0.5rem;
#         font-family: cursive; /* Change font family to cursive */
#     }
#     .bot {
#         text-align: left;
#         color: #2c3e50;
#     }
#     .user {
#         text-align: right;
#         color: #2980b9;
#     }
#     .button {
#         background-color: #3498db;
#         color: white;
#         border: none;
#         padding: 1rem 2rem;
#         font-size: 1.2rem;
#         border-radius: 5px;
#         cursor: pointer;
#     }
#     .button:hover {
#         background-color: #2980b9;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # Streamlit app layout
# st.write("""
# ### Stress Detection Chatbot
# Hi there! I'm here to help you detect stress. Go ahead and type your message below.
# """)

# # Initialize chat history
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []

# # Function to add a message to the chat history
# def add_message(user_msg, bot_msg):
#     st.session_state.chat_history.append((user_msg, bot_msg))

# # Function to delete a specific message from the chat history CSV file
# def delete_message(index):
#     if os.path.exists(CHAT_HISTORY_FILE):
#         chat_history_df = pd.read_csv(CHAT_HISTORY_FILE)
#         if 0 <= index < len(chat_history_df):
#             chat_history_df.drop(index, inplace=True)
#             chat_history_df.reset_index(drop=True, inplace=True)  # Reset index after deletion
#             chat_history_df.to_csv(CHAT_HISTORY_FILE, index=False)

# # Function to clear all chat history
# def clear_history():
#     st.session_state.chat_history = []

# # Chat interaction
# user_input = st.text_area("You:", "", height=100)
# if st.button("Send"):
#     if user_input:
#         cleaned_input = clean(user_input)
#         vectorized_input = tfidf.transform([cleaned_input]).toarray()
#         prediction = model.predict(vectorized_input)
#         result = "Stress Detected" if prediction[0] == 1 else "No Stress Detected"

#         # Add to chat history
#         add_message(user_input, result)
        
#         # Display chatbot's response in the dialogue box
#         st.text_area("Bot:", value=result, height=100, max_chars=None, key=None)

#         # Update chat history file
#         try:
#             chat_history_df = pd.read_csv(CHAT_HISTORY_FILE)
#             chat_history_df = pd.concat([chat_history_df, pd.DataFrame({'User': [user_input], 'Bot': [result]})], ignore_index=True)
#             chat_history_df.to_csv(CHAT_HISTORY_FILE, index=False)
#         except Exception as e:
#             st.error(f"An error occurred while updating the chat history: {e}")

# # Display chat history
# st.write("### Chat History")
# for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
#     delete_button_clicked = st.button(f"Delete message {i + 1}")
#     if delete_button_clicked:
#         delete_message(i)
#         st.experimental_rerun()  # Rerun the app to reflect the updated chat history
#     else:
#         st.write(f"**You**: {user_msg}")
#         st.write(f"**Bot**: {bot_msg}")

# # Button to clear all chat history
# if st.button("Clear all history"):
#     clear_history()
#     st.experimental_rerun()

    

# # Display word cloud

# # Display word cloud
# st.write("### Word Cloud of the Text Data")
# wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['text']))
# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# st.pyplot(plt)

# # Add some information about the app
# st.write("""
# #### How it works:
# - This chatbot uses a Naive Bayes classifier trained on textual data to detect stress.
# - Type any message above and click on 'Send' to see if stress is detected.
# - The model preprocesses the text by cleaning it and converting it into numerical features using TF-IDF vectorization.
# """)

# # Display performance metrics
# st.write("### Model Performance Metrics")
# st.write(f"Accuracy: {accuracy:.2f}")
# st.write(f"Precision: {precision:.2f}")
# st.write(f"Recall: {recall:.2f}")
# st.write(f"F1 Score: {f1:.2f}")

# # Confusion matrix
# fig = plt.figure(figsize=(6, 4))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# st.pyplot(fig)

# # Distribution of Stress and Non-Stress Labels
# st.write("### Distribution of Stress and Non-Stress Labels")
# label_counts = df['label'].value_counts()
# fig = px.bar(label_counts, x=label_counts.index, y=label_counts.values, labels={'index': 'Label', 'y': 'Count'}, title='Distribution of Stress and Non-Stress Labels')
# st.plotly_chart(fig)

# # Most common words in stress-related and non-stress-related texts
# st.write("### Most Common Words in Stress-Related Texts")
# stress_texts = ' '.join(df[df['label'] == 1]['text'])
# stress_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(stress_texts)
# plt.figure(figsize=(10, 5))
# plt.imshow(stress_wordcloud, interpolation='bilinear')
# plt.axis('off')
# st.pyplot(plt)

# st.write("### Most Common Words in Non-Stress-Related Texts")
# non_stress_texts = ' '.join(df[df['label'] == 0]['text'])
# non_stress_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(non_stress_texts)
# plt.figure(figsize=(10, 5))
# plt.imshow(non_stress_wordcloud, interpolation='bilinear')
# plt.axis('off')
# st.pyplot(plt)

# # Display chat history in sidebar
# st.sidebar.title("Chat History")
# try:
#     chat_history_df = pd.read_csv(CHAT_HISTORY_FILE)
#     for index, row in chat_history_df.iterrows():
#         st.sidebar.markdown('<div class="message-container">', unsafe_allow_html=True)
#         st.sidebar.markdown('<div class="message user">' + row['User'] + '</div>', unsafe_allow_html=True)
#         st.sidebar.markdown('<div class="message bot">' + row['Bot'] + '</div>', unsafe_allow_html=True)
#         st.sidebar.markdown('</div>', unsafe_allow_html=True)
# except Exception as e:
#     st.sidebar.error(f"An error occurred while loading the chat history: {e}")

# # Display chat history graph in sidebar
# st.sidebar.title("Chat History Graph")
# if not chat_history_df.empty:
#     chat_history_df['Bot'] = chat_history_df['Bot'].apply(lambda x: 1 if x == "Stress Detected" else 0)
#     fig = px.line(chat_history_df, x=chat_history_df.index, y='Bot', markers=True, labels={'index': 'Interaction', 'Bot': 'Stress (1) / No Stress (0)'}, title='Chat History: Stress Detection Over Time')
#     st.sidebar.plotly_chart(fig)

# # Footer
# st.markdown("""
#     <div style='text-align: center; padding: 1rem; font-size: 0.9rem; color: #7f8c8d;'>
#     Developed by Hassan Arzoo - Roll No: CSE204008, Guided by Dr. Abhishek Das, Associate Professor, CSE Department, Aliah University
#     </div>
# """, unsafe_allow_html=True)


# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# import nltk
# import re
# from nltk.corpus import stopwords
# import string
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px
# import joblib
# import os

# # Download stopwords
# nltk.download('stopwords')

# # Load data
# try:
#     df = pd.read_csv('stress.csv')
# except FileNotFoundError:
#     st.error("The stress.csv file was not found.")
#     st.stop()

# # Preprocess data
# stemmer = nltk.SnowballStemmer("english")
# stopword = set(stopwords.words('english'))

# def clean(text):
#     text = str(text).lower()
#     text = re.sub(r'\[.*?\]', '', text)
#     text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
#     text = re.sub(r'\w*\d\w*', '', text)
#     text = re.sub(r'[''""…]', '', text)
#     text = re.sub(r'\n', '', text)
#     text = [word for word in text.split() if word not in stopword]
#     text = ' '.join(text)
#     text = [stemmer.stem(word) for word in text.split()]
#     text = ' '.join(text)
#     return text

# df['text'] = df['text'].apply(clean)

# # Feature extraction
# tfidf = TfidfVectorizer(max_features=5000)
# X = tfidf.fit_transform(df['text']).toarray()
# y = df['label']

# # Train model
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = MultinomialNB()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# # Performance metrics
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# conf_matrix = confusion_matrix(y_test, y_pred)

# # Save model and vectorizer
# joblib.dump(model, 'naive_bayes_model.pkl')
# joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

# # Display word cloud
# st.write("### Word Cloud of the Text Data")
# wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['text']))
# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# st.pyplot(plt)

# # Add some information about the app
# st.write("""
# #### How it works:
# - This chatbot uses a Naive Bayes classifier trained on textual data to detect stress.
# - Type any message above and click on 'Send' to see if stress is detected.
# - The model preprocesses the text by cleaning it and converting it into numerical features using TF-IDF vectorization.
# """)

# # Display performance metrics
# st.write("### Model Performance Metrics")
# st.write(f"Accuracy: {accuracy:.2f}")
# st.write(f"Precision: {precision:.2f}")
# st.write(f"Recall: {recall:.2f}")
# st.write(f"F1 Score: {f1:.2f}")

# # Confusion matrix
# fig = plt.figure(figsize=(6, 4))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# st.pyplot(fig)

# # Distribution of Stress and Non-Stress Labels
# st.write("### Distribution of Stress and Non-Stress Labels")
# label_counts = df['label'].value_counts()
# fig = px.bar(label_counts, x=label_counts.index, y=label_counts.values, labels={'index': 'Label', 'y': 'Count'}, title='Distribution of Stress and Non-Stress Labels')
# st.plotly_chart(fig)

# # Most common words in stress-related and non-stress-related texts
# st.write("### Most Common Words in Stress-Related Texts")
# stress_texts = ' '.join(df[df['label'] == 1]['text'])
# stress_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(stress_texts)
# plt.figure(figsize=(10, 5))
# plt.imshow(stress_wordcloud, interpolation='bilinear')
# plt.axis('off')
# st.pyplot(plt)

# st.write("### Most Common Words in Non-Stress-Related Texts")
# non_stress_texts = ' '.join(df[df['label'] == 0]['text'])
# non_stress_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(non_stress_texts)
# plt.figure(figsize=(10, 5))
# plt.imshow(non_stress_wordcloud, interpolation='bilinear')
# plt.axis('off')
# st.pyplot(plt)

# # Footer
# st.markdown("""
#     <div style='text-align: center; padding: 1rem; font-size: 0.9rem; color: #7f8c8d;'>
#     Developed by Hassan Arzoo - Roll No: CSE204008, Guided by Dr. Abhishek Das, Associate Professor, CSE Department, Aliah University
#     </div>
# """, unsafe_allow_html=True)



# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# import nltk
# import re
# from nltk.corpus import stopwords
# import string
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px
# import joblib
# import os

# # Download stopwords
# nltk.download('stopwords')

# # Load data
# try:
#     df = pd.read_csv('stress.csv')
# except FileNotFoundError:
#     st.error("The stress.csv file was not found.")
#     st.stop()

# # Preprocess data
# stemmer = nltk.SnowballStemmer("english")
# stopword = set(stopwords.words('english'))

# def clean(text):
#     text = str(text).lower()
#     text = re.sub(r'\[.*?\]', '', text)
#     text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
#     text = re.sub(r'\w*\d\w*', '', text)
#     text = re.sub(r'[''""…]', '', text)
#     text = re.sub(r'\n', '', text)
#     text = [word for word in text.split() if word not in stopword]
#     text = ' '.join(text)
#     text = [stemmer.stem(word) for word in text.split()]
#     text = ' '.join(text)
#     return text

# df['text'] = df['text'].apply(clean)

# # Feature extraction
# tfidf = TfidfVectorizer(max_features=5000)
# X = tfidf.fit_transform(df['text']).toarray()
# y = df['label']

# # Train model
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = MultinomialNB()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# # Performance metrics
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# conf_matrix = confusion_matrix(y_test, y_pred)

# # Save model and vectorizer
# joblib.dump(model, 'naive_bayes_model.pkl')
# joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

# # Display word cloud
# st.write("### Word Cloud of the Text Data")
# wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['text']))
# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# st.pyplot(plt)

# # Add some information about the app
# st.write("""
# #### How it works:
# - This chatbot uses a Naive Bayes classifier trained on textual data to detect stress.
# - Type any message above and click on 'Send' to see if stress is detected.
# - The model preprocesses the text by cleaning it and converting it into numerical features using TF-IDF vectorization.
# """)

# # Display performance metrics
# st.write("### Model Performance Metrics")
# st.write(f"Accuracy: {accuracy:.2f}")
# st.write(f"Precision: {precision:.2f}")
# st.write(f"Recall: {recall:.2f}")
# st.write(f"F1 Score: {f1:.2f}")

# # Confusion matrix
# fig = plt.figure(figsize=(6, 4))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# st.pyplot(fig)

# # Distribution of Stress and Non-Stress Labels
# st.write("### Distribution of Stress and Non-Stress Labels")
# label_counts = df['label'].value_counts()
# fig = px.bar(label_counts, x=label_counts.index, y=label_counts.values, labels={'index': 'Label', 'y': 'Count'}, title='Distribution of Stress and Non-Stress Labels')
# st.plotly_chart(fig)

# # Most common words in stress-related and non-stress-related texts
# st.write("### Most Common Words in Stress-Related Texts")
# stress_texts = ' '.join(df[df['label'] == 1]['text'])
# stress_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(stress_texts)
# plt.figure(figsize=(10, 5))
# plt.imshow(stress_wordcloud, interpolation='bilinear')
# plt.axis('off')
# st.pyplot(plt)

# st.write("### Most Common Words in Non-Stress-Related Texts")
# non_stress_texts = ' '.join(df[df['label'] == 0]['text'])
# non_stress_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(non_stress_texts)
# plt.figure(figsize=(10, 5))
# plt.imshow(non_stress_wordcloud, interpolation='bilinear')
# plt.axis('off')
# st.pyplot(plt)

# # Footer
# st.markdown("""
#     <div style='text-align: center; padding: 1rem; font-size: 0.9rem; color: #7f8c8d;'>
#     Developed by Hassan Arzoo - Roll No: CSE204008, Guided by Dr. Abhishek Das, Associate Professor, CSE Department, Aliah University
#     </div>
# """, unsafe_allow_html=True)


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import nltk
import re
from nltk.corpus import stopwords
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
import os

# Download stopwords
nltk.download('stopwords')

# Load data
try:
    df = pd.read_csv('stress.csv')
except FileNotFoundError:
    st.error("The stress.csv file was not found.")
    st.stop()

# Preprocess data
stemmer = nltk.SnowballStemmer("english")
stopword = set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'[''""…]', '', text)
    text = re.sub(r'\n', '', text)
    text = [word for word in text.split() if word not in stopword]
    text = ' '.join(text)
    text = [stemmer.stem(word) for word in text.split()]
    text = ' '.join(text)
    return text

df['text'] = df['text'].apply(clean)

# Feature extraction
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['text']).toarray()
y = df['label']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Save model and vectorizer
joblib.dump(model, 'naive_bayes_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

# Initialize chat history file
CHAT_HISTORY_FILE = 'chat_history.csv'
if not os.path.exists(CHAT_HISTORY_FILE):
    pd.DataFrame(columns=['User', 'Bot']).to_csv(CHAT_HISTORY_FILE, index=False)

# Custom CSS for styling
st.markdown("""
    <style>
    .message-container {
        margin-bottom: 1rem;
    }
    .message {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        font-family: cursive; /* Change font family to cursive */
    }
    .bot {
        text-align: left;
        color: #2c3e50;
    }
    .user {
        text-align: right;
        color: #2980b9;
    }
    .button {
        background-color: #3498db;
        color: white;
        border: none;
        padding: 1rem 2rem;
        font-size: 1.2rem;
        border-radius: 5px;
        cursor: pointer;
    }
    .button:hover {
        background-color: #2980b9;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit app layout
st.write("""
### Stress Detection Chatbot
Hi there! I'm here to help you detect stress. Go ahead and type your message below.
""")

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function to add a message to the chat history
def add_message(user_msg, bot_msg):
    st.session_state.chat_history.append((user_msg, bot_msg))

# Chat interaction
user_input = st.text_area("You:", "", height=100)
if st.button("Send"):
    if user_input:
        cleaned_input = clean(user_input)
        vectorized_input = tfidf.transform([cleaned_input]).toarray()
        prediction = model.predict(vectorized_input)
        result = "Stress Detected" if prediction[0] == 1 else "No Stress Detected"

        # Add to chat history
        add_message(user_input, result)
        
        # Display chatbot's response in the dialogue box
        st.text_area("Bot:", value=result, height=100, max_chars=None, key=None)

        # Update chat history file
        try:
            chat_history_df = pd.read_csv(CHAT_HISTORY_FILE)
            chat_history_df = pd.concat([chat_history_df, pd.DataFrame({'User': [user_input], 'Bot': [result]})], ignore_index=True)
            chat_history_df.to_csv(CHAT_HISTORY_FILE, index=False)
        except Exception as e:
            st.error(f"An error occurred while updating the chat history: {e}")

# Display chat history
st.write("### Chat History")
for user_msg, bot_msg in st.session_state.chat_history:
    st.write(f"**You**: {user_msg}")
    st.write(f"**Bot**: {bot_msg}")

# Display word cloud
st.write("### Word Cloud of the Text Data")
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['text']))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
st.pyplot(plt)

# Add some information about the app
st.write("""
#### How it works:
- This chatbot uses a Naive Bayes classifier trained on textual data to detect stress.
- Type any message above and click on 'Send' to see if stress is detected.
- The model preprocesses the text by cleaning it and converting it into numerical features using TF-IDF vectorization.
""")

# Display performance metrics
st.write("### Model Performance Metrics")
st.write(f"Accuracy: {accuracy:.2f}")
st.write(f"Precision: {precision:.2f}")
st.write(f"Recall: {recall:.2f}")
st.write(f"F1 Score: {f1:.2f}")

# Confusion matrix
fig = plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
st.pyplot(fig)

# Distribution of Stress and Non-Stress Labels
st.write("### Distribution of Stress and Non-Stress Labels")
label_counts = df['label'].value_counts()
fig = px.bar(label_counts, x=label_counts.index, y=label_counts.values, labels={'index': 'Label', 'y': 'Count'}, title='Distribution of Stress and Non-Stress Labels')
st.plotly_chart(fig)

# Most common words in stress-related and non-stress-related texts
st.write("### Most Common Words in Stress-Related Texts")
stress_texts = ' '.join(df[df['label'] == 1]['text'])
stress_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(stress_texts)
plt.figure(figsize=(10, 5))
plt.imshow(stress_wordcloud, interpolation='bilinear')
plt.axis('off')
st.pyplot(plt)

st.write("### Most Common Words in Non-Stress-Related Texts")
non_stress_texts = ' '.join(df[df['label'] == 0]['text'])
non_stress_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(non_stress_texts)
plt.figure(figsize=(10, 5))
plt.imshow(non_stress_wordcloud, interpolation='bilinear')
plt.axis('off')
st.pyplot(plt)

# Display chat history in sidebar
st.sidebar.title("Chat History")
try:
    chat_history_df = pd.read_csv(CHAT_HISTORY_FILE)
    for index, row in chat_history_df.iterrows():
        st.sidebar.markdown('<div class="message-container">', unsafe_allow_html=True)
        st.sidebar.markdown('<div class="message user">' + row['User'] + '</div>', unsafe_allow_html=True)
        st.sidebar.markdown('<div class="message bot">' + row['Bot'] + '</div>', unsafe_allow_html=True)
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
except Exception as e:
    st.sidebar.error(f"An error occurred while loading the chat history: {e}")

# Display chat history graph in sidebar
st.sidebar.title("Chat History Graph")
if not chat_history_df.empty:
    chat_history_df['Bot'] = chat_history_df['Bot'].apply(lambda x: 1 if x == "Stress Detected" else 0)
    fig = px.line(chat_history_df, x=chat_history_df.index, y='Bot', markers=True, labels={'index': 'Interaction', 'Bot': 'Stress (1) / No Stress (0)'}, title='Chat History: Stress Detection Over Time')
    st.sidebar.plotly_chart(fig)

# Footer
st.markdown("""
    <div style='text-align: center; padding: 1rem; font-size: 0.9rem; color: #7f8c8d;'>
    Developed by Hassan Arzoo - Roll No: CSE204008, Guided by Dr. Abhishek Das, Associate Professor, CSE Department, Aliah University
    </div>
""", unsafe_allow_html=True)

