import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import datetime as dt
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow import keras
from keras.models import model_from_json
import random
import re
import unicodedata
from nltk.corpus import stopwords
from matplotlib.image import imread
import tensorflow as tf
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

st.session_state.image_predicted_class_title = ''
st.session_state.texte_predicted_class_title = ''
# st.session_state.image_model: tf.keras.Sequential = None
# = None
# st.session_state.image_model: tf.keras.Sequential = ''

st.title('Projet Pykuten V2bis')

st.write(
    'Ce projet se déroule à la fois dans le cadre de notre formation avec DataScientest et d’un concours organisé par Rakuten Institute of Technology Paris.: Rakuten France Multimodal Product Data Classification.')

stop_words = stopwords.words(['french', 'english'])


def load_text_model(model=''):
    print(model)
    text_model_ = keras.models.load_model("./Modeles/" + model + ".h5")
    return text_model_


def load_image_model(model=''):
    model_json = './Modeles/' + model + '.json'
    model_h5 = './Modeles/' + model + '.h5'

    with open(model_json, 'r') as fx:
        model_json_string = fx.read()

    image_model_ = model_from_json(model_json_string)
    image_model_.load_weights(model_h5)
    return image_model_


def load_data_test():
    xx_df = pd.read_csv("./Modeles/data_test_csv.csv")
    return xx_df.iloc[:, 1:]


def load_X_train():
    return pd.read_csv('./Modeles/X_train_csv.csv', index_col=0)


def load_X_test():
    return pd.read_csv('./Modeles/X_test_csv.csv', index_col=0)


def load_product_catalog():
    return pd.read_csv('./Data_sources/classes produits.csv', index_col=0)


def load_y_train():
    return pd.read_csv('./Data_sources/Y_train.csv', index_col=0)


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!]+", " ", w)
    w = re.sub(r'\b\w{0,2}\b', '', w)

    # remove stopword
    mots = word_tokenize(w.strip())
    mots = [mot for mot in mots if mot not in stop_words]

    return ' '.join(mots).strip()


def format_pourcentage(text):
    point = text.index('.')
    return text[0:(point+3)]


# Product_catalog=pd.read_csv('Data_sources/classes produits.csv', index_col=0)
select_modele_text = st.sidebar.selectbox('Modèle de classification de textes', options=['---', 'RNN(LSTM, lr=0.001)'])
select_modele_image = st.sidebar.selectbox("Modèle de classification d'images", options=['---', 'ResNet50(lr=0.001)'])


def load_models():
    print("-----------------load_models------------------------")

    if select_modele_text == '---':
        st.sidebar.write('Modèle de classification de textes invalide')
        return

    if select_modele_image == '---':
        st.sidebar.write("Modèle de classification d'images invalide")
        return

    # Suite... ####
    if select_modele_text == 'RNN(LSTM, lr=0.001)':
        st.session_state.text_model = load_text_model('64_0.001_textClassifierRNNLSTMBow')

    if select_modele_image == 'ResNet50(lr=0.001)':
        st.session_state.image_model = load_image_model('64_0.001_classifierTranferLearningResNet50')

    st.session_state.Product_catalog = load_product_catalog()
    st.session_state.y_train = load_y_train()

    dict_code_to_id = {}
    dict_id_to_code = {}

    list_tags = list(st.session_state.y_train['prdtypecode'].unique())

    for i, tag in enumerate(list_tags):
        dict_code_to_id[tag] = i
        dict_id_to_code[i] = tag

    st.session_state.data_test = load_data_test()
    st.session_state.data_test = st.session_state.data_test.iloc[:, 1:]
    st.session_state.data_test['label'] = st.session_state.data_test['label'].astype('string')

    st.session_state.dict_code_to_id = dict_code_to_id
    st.session_state.dict_id_to_code = dict_id_to_code

    st.session_state.x_train = load_X_train()
    st.session_state.x_train = st.session_state.x_train.iloc[:, 1:]
    st.session_state.x_train['label'] = st.session_state.x_train['label'].astype('string')
    st.session_state.x_train['text'] = st.session_state.x_train['text'].astype('string')

    st.session_state.x_train.text = st.session_state.x_train.text.apply(
        lambda x: x if isinstance(x, str) == True else "")

    # Définition du tokenizer
    st.session_state.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
    # Mettre à jour le dictionnaire du tokenizer
    st.session_state.tokenizer.fit_on_texts(st.session_state.x_train.text)

    st.session_state.x_test = load_X_test()
    st.session_state.x_test = st.session_state.x_test.iloc[:, 1:]
    st.session_state.x_test['label'] = st.session_state.x_test['label'].astype('string')
    st.session_state.x_test['text'] = st.session_state.x_test['text'].astype('string')

    return


if st.sidebar.button("Charger les modèles"):
    load_models()


def predict_image(data_test_one, img_model, target_size=(256, 256), batch_size=64):
    dict_classes = {'0': 0,
                    '1': 1,
                    '10': 2,
                    '11': 3,
                    '12': 4,
                    '13': 5,
                    '14': 6,
                    '15': 7,
                    '16': 8,
                    '17': 9,
                    '18': 10,
                    '19': 11,
                    '2': 12,
                    '20': 13,
                    '21': 14,
                    '22': 15,
                    '23': 16,
                    '24': 17,
                    '25': 18,
                    '26': 19,
                    '3': 20,
                    '4': 21,
                    '5': 22,
                    '6': 23,
                    '7': 24,
                    '8': 25,
                    '9': 26}

    # print(st.session_state.data_test.info())
    #data_test_df = st.session_state.data_test.loc[[index]]
    #data_test_df = st.session_state.x_test.loc[[index]]

    data_test_df = data_test_one

    # print(data_test_df.head())

    # Configure ImageDataGenerator
    img_gen_test = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    data_flow_test_one = img_gen_test.flow_from_dataframe(
        dataframe=data_test_df,
        target_size=target_size,
        shuffle=False,
        x_col='path',  # column containing path to image
        y_col='label',  # column containing label
        class_mode='sparse',  # classes are not one-hot encoded
        batch_size=batch_size
    )

    reconstructed_y_prob_one = img_model.predict(
        x=data_flow_test_one,
        batch_size=None,  # specified by generator
        steps=None,  # specified by generator
        verbose=1
    )

    # print("reconstructed_y_prob_one: ", reconstructed_y_prob_one)

    # ATTENTION, tour de passe-passe pour s'en tirer
    # ATTENTION: Besoin de créer le tmp_dict et d'exécuter le data_flow_test pour récupérer l'intégralité des classes

    # print("reconstructed_y_prob_one")
    # print(reconstructed_y_prob_one)

    predicted_class = np.argmax(reconstructed_y_prob_one, axis=1)[0]

    # print("predicted_class")
    # print(predicted_class)

    for key, val in dict_classes.items():
        if predicted_class == val:
            final_predicted = key

    real_class = int(data_test_df['label'].values[0])

    # st.write("Classe réelle: ", real_class, '\t',
    #         st.session_state.Product_catalog[st.session_state.Product_catalog.index == st.session_state.dict_id_to_code[real_class]]['prdtypecategory'].values[0])
    # st.write("Classe prédite: ", final_predicted, '\t',
    #         st.session_state.Product_catalog[st.session_state.Product_catalog.index == st.session_state.dict_id_to_code[int(final_predicted)]]['prdtypecategory'].values[
    #             0])

    st.session_state.image_real_class_title = st.session_state.Product_catalog[
        st.session_state.Product_catalog.index == st.session_state.dict_id_to_code[real_class]][
        'prdtypecategory'].values[0]
    st.session_state.image_predicted_class_title = st.session_state.Product_catalog[
        st.session_state.Product_catalog.index == st.session_state.dict_id_to_code[int(final_predicted)]][
        'prdtypecategory'].values[
        0]

    im_path = data_test_df['path'].values[0]
    st.session_state.image_predicted_class_path = im_path

    st.session_state.image_predicted_class_percentage = reconstructed_y_prob_one[0, predicted_class]

    # st.write("im_path\n")
    # st.write(im_path)

    # plt.imshow(imread(im_path))
    # plt.axis('off');
    # plt.show()

    return final_predicted, reconstructed_y_prob_one[0, predicted_class]


def predict_text(data_test_one, txt_model, maxlen=48):
    # Nettoyage avant calcul
    data_test_one.text = data_test_one.text.apply(lambda x: preprocess_sentence(x))
    sequences_one = st.session_state.tokenizer.texts_to_sequences(data_test_one.text)
    # maxlen=maxlen,
    sequences_one = tf.keras.preprocessing.sequence.pad_sequences(sequences_one, padding='post')

    y_text_pred_one = txt_model.predict(sequences_one)

    # print("y_text_pred_one")
    # print(y_text_pred_one)

    predicted_one_class = np.argmax(y_text_pred_one, axis=1)[0]

    # print("predicted_one_class")
    # print(predicted_one_class)

    dict_classes = {'0': 0,
                    '1': 1,
                    '10': 2,
                    '11': 3,
                    '12': 4,
                    '13': 5,
                    '14': 6,
                    '15': 7,
                    '16': 8,
                    '17': 9,
                    '18': 10,
                    '19': 11,
                    '2': 12,
                    '20': 13,
                    '21': 14,
                    '22': 15,
                    '23': 16,
                    '24': 17,
                    '25': 18,
                    '26': 19,
                    '3': 20,
                    '4': 21,
                    '5': 22,
                    '6': 23,
                    '7': 24,
                    '8': 25,
                    '9': 26}

    for key, val in dict_classes.items():
        if predicted_one_class == val:
            final_predicted = key

    st.session_state.texte_predicted_class_title = st.session_state.Product_catalog[
        st.session_state.Product_catalog.index == st.session_state.dict_id_to_code[int(predicted_one_class)]][
        'prdtypecategory'].values[
        0]

    st.session_state.text_to_predict = data_test_one.text.iloc[0]

    st.session_state.texte_predicted_class_percentage = y_text_pred_one[0, predicted_one_class]
    return final_predicted, y_text_pred_one[0, predicted_one_class]


def predict_final():
    index = np.random.choice(st.session_state.data_test.index)
    data_test_df = st.session_state.data_test.loc[[index]]

    #index = np.random.choice(st.session_state.x_test.index)
    #data_test_df = st.session_state.x_test.loc[[index]]

    print("INDEX: ", index)
    print(data_test_df)

    image_predicted, image_percentage = predict_image(data_test_df, st.session_state.image_model)
    text_predicted, text_percentage = predict_text(data_test_df, st.session_state.text_model)

    print("image_predicted: ", image_predicted)
    print("image_percentage: ", image_percentage)

    print("text_predicted: ", text_predicted)
    print("text_percentage: ", text_percentage)

    print("st.session_state.dict_id_to_code: \n", st.session_state.dict_id_to_code)
    print("st.session_state.Product_catalog: \n", st.session_state.Product_catalog)


if st.sidebar.button("Get Item"):
    predict_final()

#######
col1, col2 = st.columns(2)

with col1:
    st.write('Image')
    print(st.session_state.image_predicted_class_title)
    if st.session_state.image_predicted_class_title != '':
        image = Image.open(st.session_state.image_predicted_class_path)
        st.image(image, width=150, caption=st.session_state.image_predicted_class_title)
        st.write('Classe réelle: ', st.session_state.image_real_class_title)
        st.write('Classe prédite: ', st.session_state.image_predicted_class_title)
        #st.write('(%)): ', round(st.session_state.image_predicted_class_percentage * 100, 2))
        st.write('(%)): ', format_pourcentage(str(st.session_state.image_predicted_class_percentage * 100)))

with col2:
    st.write('Texte')
    if st.session_state.texte_predicted_class_title != '':
        st.write('Texte: ', st.session_state.text_to_predict)

        st.write('Classe prédite: ', st.session_state.texte_predicted_class_title)
        #st.write('(%)): ', round(st.session_state.texte_predicted_class_percentage * 100, 2))
        st.write('(%)): ', format_pourcentage(str(st.session_state.texte_predicted_class_percentage * 100)))

        st.text_area("string xxxx stringstringstring stringstringstringstringstringstringstringstringstringstringstringstringstringstringstring")


st.write('Prédiction finale')
