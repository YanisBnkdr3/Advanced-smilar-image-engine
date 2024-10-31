import streamlit as st
import requests
import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Configuration de Streamlit
st.set_page_config(page_title='Moteur de Recherche d\'Images', layout='wide')

def main():
    st.title("Moteur de Recherche d'Images")

    # Formulaire de téléversement
    st.sidebar.header("Téléversez une Image")
    uploaded_file = st.sidebar.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

    # Sélecteurs pour le descripteur et la distance
    descriptor_choice = st.sidebar.selectbox("Choisissez le descripteur", ["GLCM", "BitDesc"])
    distance_choice = st.sidebar.selectbox("Choisissez la distance", ["Manhattan", "Canberra", "Euclidean", "Chebyshev"])
    algorithm_choice = st.sidebar.selectbox("Choisissez l'algorithme", ["LDA", "SVC", "SVR"])

    
    # Sélecteur pour le nombre d'images à afficher
    num_images = st.sidebar.slider("Sélectionnez le nombre d'images à afficher", min_value=1, max_value=50, value=10)

    if uploaded_file:
        # Sauvegarder l'image téléversée
        image_path = os.path.join('static/uploads', uploaded_file.name)
        with open(image_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        st.image(image_path, caption='Image Téléversée', use_column_width=True)
        
        # Effectuer la recherche
        if st.sidebar.button('Rechercher des Images Similaires'):
            data = {'descriptor': descriptor_choice, 'distance': distance_choice,'top_k':num_images,'algorithm': algorithm_choice}
            files = {'file': open(image_path, 'rb')}
            response = requests.post("http://127.0.0.1:5001/api/search", data=data, files=files)
            
            if response.status_code == 200:
                results = response.json()
                
                # Afficher les images similaires
                st.header(f"Top {num_images} Images Similaires")
                if 'similar_images' in results:
                    similar_images = results['similar_images'][:num_images]
                    
                    # Organiser les images en colonnes
                    cols = st.columns(5)  # Ajustez le nombre de colonnes selon vos besoins
                    for idx, result in enumerate(similar_images):
                        col = cols[idx % 5]
                        col.image(result['image_path'], caption=f"Distance: {result['distance']}", width=150)  # Fixez la largeur à 150 pixels
                    
                # Afficher la distribution des classes
                st.header("Distribution des Classes")
                if 'class_distribution' in results:
                    class_distribution = results['class_distribution']
                    class_distribution_list = [(k, v) for k, v in class_distribution.items()]
                    st.write(pd.DataFrame(class_distribution_list, columns=['Classe', 'Nombre']))

                    # Charger le modèle
                    with open('model.pkl', 'rb') as f:
                        model = pickle.load(f)
                    
                    # Afficher le diagramme
                    labels = list(class_distribution.keys())
                    counts = list(class_distribution.values())

                    fig, ax = plt.subplots()
                    ax.bar(labels, counts)
                    ax.set_xlabel('Classe')
                    ax.set_ylabel('Nombre de photos similaires')
                    ax.set_title('Distribution des classes des photos similaires')
                    st.pyplot(fig)
            else:
                st.write(f"Erreur: {response.status_code}")
                st.write(response.text)

if __name__ == "__main__":
    main()
