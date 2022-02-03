import streamlit as st
import cv2
import numpy as np
def main():
    """Streamlit presentation: Projet Pykuten"""
    
    # Create sidebar and set title
    st.title("Projet Pykuten")
    st.write("""
    ## Presentation d'un projet de classification multimodal
    """
    )

    ### Initialise session states:
    if 'img_model_loaded' not in st.session_state:
        st.session_state.img_model_loaded=False

    if 'txt_model_loaded' not in st.session_state:
        st.session_state.txt_model_loaded=False

    if 'article_loaded' not in st.session_state:
        st.session_state.article_loaded=False

    st.session_state.model_loaded=st.session_state.img_model_loaded*st.session_state.txt_model_loaded
    
    img_col, txt_col=st.columns([2,1])

    ###### Chargement des modeles:

    ### Modele image
    img_col.header("Image")
    img_param=img_col.container()
    img_param.subheader("Paramètres")
    img_model_name= img_param.radio("Select img base model", ("VGG16","ResNet50","MobilNetV2"))
    img_model_LR= img_param.radio("Select img model Learning rate", ("0.01","0.001"))
    load_img_model_btn=img_param.button('Load Model', key='img_model_btn')

    def load_img_model(name,LR):
        model_found=False

        if model_found:
            st.session_state.img_model_loaded=True
            st.session_state.img_model_name=name
            st.session_state.img_model_LR=LR
        else:
            st.session_state.img_model_loaded=False
            img_param.write('Image model loading function not implemented')
    
    if load_img_model_btn:        
        load_img_model(img_model_name, img_model_LR)

    if st.session_state.img_model_loaded:
        # model_name='Model:{0}LR:{1}'.format(st.session_state.img_model_name,st.session_state.img_model_LR)
        img_param.write('Model : **{0}**'.format(st.session_state.img_model_name))
        img_param.write('LR    : **{0}**'.format(st.session_state.img_model_LR))

    ### Modele Text
    txt_col.header("Texte")
    txt_param=txt_col.container()

    txt_param.subheader("Paramètres")
    txt_model_name= txt_param.radio("Select txt base model", ("GRU","LSTM"))
    txt_model_LR= txt_param.radio("Select txt model Learning rate", ("0.01","0.001"))
    load_txt_model_btn=txt_param.button('Load Model',key='txt_model_btn')


    def load_txt_model(name,LR):
        txt_param.write('Text model loading function not implemented')
        st.session_state.txt_model_loaded=True
        st.session_state.txt_model_name=name
        st.session_state.txt_model_LR=LR

    if load_txt_model_btn:
        load_txt_model(txt_model_name,txt_model_LR)
    
    if st.session_state.txt_model_loaded:
        txt_param.write('Model : **{0}**'.format(st.session_state.txt_model_name))
        txt_param.write('LR    : **{0}**'.format(st.session_state.txt_model_LR))


    ###### Chargement de l'article
    article=st.container()
    article.write("""# ARTICLE""")
    load_article_btn=article.button('Load Article', key='Load_Article_btn')
    article_img, article_txt = article.columns(2)

    #### PREDICTION

    predict=st.container()
    predict.write("""# PREDICTION""")
    predict_img, predict_txt, predict_final, real_class = predict.columns(4)

    def load_article():
        img=cv2.imread(r'Data_sources\Images\test_image.png')
        txt="Txt Article Chargé"
        return(img, txt)

    def get_img_prediction(img):
        pred='Classe prédite image'
        prct= np.random.rand()
        return pred, prct
    
    def get_txt_prediction(img):
        pred='Classe prédite Text'
        prct= np.random.rand()
        return pred, prct

    def get_final_prediction(img,txt):
        pred='Classe prédite Full'
        prct= np.random.rand()
        return pred, prct

    def get_real_class(index):
        return 'Classe Réelle'
    
    def format_html_markdown(txt,txt_color):    
        return '<p style="font-family:sans-serif; color:'+ txt_color +'; font-size: 25px;">'+ txt +'</p>' 

    if load_article_btn:

        img, txt=load_article()
        article_img.image(img,channels ='BGR')
        article_txt.write(txt)
        st.session_state.article_loaded=True

        if st.session_state.model_loaded:
            img_pred,img_score=get_img_prediction(img)
            txt_pred,txt_score=get_txt_prediction(txt)
            final_pred,final_score=get_final_prediction(img,txt)
            true_class=get_real_class(0)

            predict_img.write("### Image")
            
            # if img_pred==true_class:
            if img_score>0.5:
                tmp_color='green'
            else:
                tmp_color='red'
            
            predict_img.markdown(format_html_markdown(img_pred,tmp_color), unsafe_allow_html=True)
            predict_img.markdown(format_html_markdown('{:.1f}%'.format(img_score*100),tmp_color), unsafe_allow_html=True)

            predict_txt.write("### Text")


            # if txt_pred==true_class:
            if txt_score>0.5:
                tmp_color='green'
            else:
                tmp_color='red'
            
            predict_txt.markdown(format_html_markdown(txt_pred,tmp_color), unsafe_allow_html=True)
            predict_txt.markdown(format_html_markdown('{:.1f}%'.format(txt_score*100),tmp_color), unsafe_allow_html=True)

            predict_final.write("### Full Model")

            # if final_pred==true_class:
            if final_score>0.5:
                tmp_color='green'
            else:
                tmp_color='red'
            
            predict_final.markdown(format_html_markdown(final_pred,tmp_color), unsafe_allow_html=True)
            predict_final.markdown(format_html_markdown('{:.1f}%'.format(final_score*100),tmp_color), unsafe_allow_html=True)

            real_class.write("### Réelle")
            real_class.markdown(format_html_markdown(true_class,'blue'), unsafe_allow_html=True)

        else:
            predict.write("## Model non chargé")


if __name__ == "__main__":
    main()