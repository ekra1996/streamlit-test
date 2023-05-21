import streamlit as st
import numpy as np 
import pandas as pd 
import joblib 
import streamlit as st
#from PIL import Image
from deep_translator import GoogleTranslator


# WEB APP  
st.write("Bienvenue !  \nCeci est une application créée dans le cadre d'un projet ESG, avec pour but de déterminer l'éco-responsabilité d'un vêtement en fonction de sa composition principale.")
st.title("Votre Vêtement est-il éco-responsable ? :jeans: :necktie: :dress: ")
st.subheader("Voyons-voir comment l'Intelligence Artificielle nous permet de répondre.")

#image = Image.open('eco_fashion.jpg')
#st.image(image, caption='Eco Fashion.')



# Deployement
model = joblib.load(filename="final_model.joblib")

materials = [
 'acrylic',
 'camel',
 'copper',
 'cotton',
 'elastane',
 'lana',
 'lino',
 'liocel',
 'metallic fiber',
 'modal',
 'nylon',
 'polyamide',
 'polyester',
 'viscose',
 ]

compo = ["material_"+i for i in materials]
columns=compo
columns.insert(0,'percent')


st.sidebar.header("1. Choissisez le composant principal.")
material = st.sidebar.selectbox('Composant',compo[1:])

st.sidebar.header("2 .Quel est le pourcentage ?")
percent = st.sidebar.slider(" Pourcentage ", 0.00,1.00)

material_name = material.split('_')[1]
st.sidebar.write("Votre article est composé de ", GoogleTranslator(source='en', target='fr').translate(material_name)
 ," à ", int(percent*100) ,"%")

material_acrylic=material 
material_camel = material
material_copper= material
material_cotton= material
material_elastane= material
material_lana= material
material_lino=material
material_liocel= material
material_metallic_fiber= material
material_modal= material
material_nylon= material
material_polyamide= material
material_polyester= material
material_viscose= material


row = np.array([percent,material_acrylic,material_camel,material_copper,material_cotton,material_elastane,
material_lana,material_lino,material_liocel,material_metallic_fiber,material_modal,
material_nylon,material_polyamide,material_polyester,material_viscose]).reshape(1,-1)

X = pd.DataFrame(data = row,columns=columns)

X[columns[1:]]=X.columns[1:].map(lambda x : 1 if x == material else 0 )

st.subheader(" 3. Récapitulatif ")
st.write("Nous allons maintenant faire la prédiction à partir des informations contenues dans le tableau suivant : ")
st.dataframe(X)
st.write("*NB : L'indication 0/1 fait référence à la composition de votre article.*")

st.subheader("4 .C'est parti pour la prédiction.")
# Prediction 
def predict():
    prediction = model.predict(X)[0]
    
    if prediction == 1:
        st.success(""" Bravo !, ce vêtement est éco-responsable. :thumbsup:.  \nAvant de partir, permettez-nous de vous remercier pour votre contribution à la préservation de notre planète.""")
    else : 
        st.error(""" Malheureusement, ce vêtement n'est pas éco-responsable. :thumbsdown:  \nNous vous recommandons l'usage de vêtements faits à base de matières   \nbio-dégradables telles que : Le coton, le lin, la fibre métalisée ou encore la laine.  \n:exclamation: Toutefois, l'usage abusif de ces matières pourrait entrainer des effets néfastes pour l'environnement.""")
  
st.button(':green[Predict]', on_click=predict) 

st.write(":point_up_2: Cliquez pour prédire.  ")


st.markdown("<h5 style='text-align: center;'>La réponse au haut de la page.", unsafe_allow_html=True)


st.markdown("<h1 style='text-align: center;'>Merci !</h1>", unsafe_allow_html=True)

st.write("Hello word")
