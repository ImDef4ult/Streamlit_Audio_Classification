import streamlit as st
from PIL import Image
from utils import (
    draw_spectrogram,
    extract_features,
    predict
)


def main():
    # --------------------------------
    # Side Bar
    # --------------------------------
    Side = st.sidebar
    Side.header('Try it out!!')
    user_file = Side.file_uploader('Upload your audio file')

    # Banner Image
    imageHome = Image.open('Resources/Banner.jpg')
    st.image(imageHome, use_column_width=True)
    st.title('Audio Recognition')

    if not user_file:
        About = st.expander("Acerca de")
        About.markdown("""
            * **Autores:** Daniel Andres Jimenez, Alejandro Leon Andrade, Andres Felipe Rojas
            * **Materia:** Deep Learning
            * **Librerias utilizadas:** pandas, streamlit, PIL, Keras, librosa
            * **Fuente de datos:** [UrbanSounds8k](https://urbansounddataset.weebly.com/urbansound8k.html).
            """)
        st.header('Planteamiento del problema')
        st.write('''
            Dados los bajos costes en sensores, y teniendo en cuenta la vida en grandes ciudades como Bogotá, 
            es interesante identificar los diferentes sonidos urbanos presentes en toda la ciudad. 
            Al tener identificadas diferentes zonas de la ciudad con cierto tipo de sonidos, instituciones como 
            La secretaria Distrital de Ambiente, pueden tomar decisiones concretas y rápidas, respecto a la 
            contaminación ambiental presentada. En futuros trabajos es posible combinar la identificación del 
            sonido junto con los niveles de intensidad del sonido, para tomar mejores decisiones.
        ''')

        st.header('Objetivo de negocio')
        st.write('''
            Realizar un modelo de aprendizaje profundo que sea capaz de identificar y clasificar sonidos urbanos en 10 
            categorías iniciales y básicas de la taxonomía de sonidos urbanos.
        ''')

        st.header('Como son los datos?')
        air, car, kids, dog = st.columns(4)
        with air:
            # Air conditioner
            st.subheader('Aire acondicionado')

            air_image = Image.open('Resources/air_conditioner.jpg')
            air.image(air_image, use_column_width=True)

            file_path = 'samples/air.wav'
            air.pyplot(draw_spectrogram(file_path))
            st.audio(file_path)
        with car:
            # Car horn
            st.subheader('Bocina de auto')

            car_image = Image.open('Resources/car_horn.jpg')
            car.image(car_image, use_column_width=True)

            file_path = 'samples/car.wav'
            car.pyplot(draw_spectrogram(file_path))
            st.audio(file_path)

        with kids:
            # Children playing
            st.subheader('Niños jugando')
            kid_image = Image.open('Resources/children_playing.jpg')
            kids.image(kid_image, use_column_width=True)

            file_path = 'samples/children.wav'
            kids.pyplot(draw_spectrogram(file_path))
            st.audio(file_path)

        with dog:
            # Dog bark
            st.subheader('Perro ladrando')

            dog_image = Image.open('Resources/dog_bark.jpg')
            dog.image(dog_image, use_column_width=True)

            file_path = 'samples/dog.wav'
            dog.pyplot(draw_spectrogram(file_path))
            st.audio(file_path)

        drill, eng, gun, ham = st.columns(4)
        with drill:
            # Drilling
            st.subheader('Taladro')

            car_image = Image.open('Resources/drilling.jpg')
            drill.image(car_image, use_column_width=True)

            file_path = 'samples/drilling.wav'
            drill.pyplot(draw_spectrogram(file_path))
            st.audio(file_path)

        with eng:
            # Engine Idling
            st.subheader('Encendiendo motor')
            engine_image = Image.open('Resources/engine_idling.jpg')
            eng.image(engine_image, use_column_width=True)

            file_path = 'samples/engine.wav'
            eng.pyplot(draw_spectrogram(file_path))
            st.audio(file_path)

        with gun:
            # Gun shot
            st.subheader('Disparo de arma')

            gun_image = Image.open('Resources/gun_shot.jpg')
            gun.image(gun_image, use_column_width=True)

            file_path = 'samples/gun.wav'
            gun.pyplot(draw_spectrogram(file_path))
            st.audio(file_path)

        with ham:
            # Jackhammer
            st.subheader('Martillo neumático')

            hammer_image = Image.open('Resources/jackhammer.jpg')
            ham.image(hammer_image, use_column_width=True)

            file_path = 'samples/hammer.wav'
            ham.pyplot(draw_spectrogram(file_path))
            st.audio(file_path)

        siren, street = st.columns(2)
        with siren:
            # Siren
            st.subheader('Sirena')
            siren_image = Image.open('Resources/siren.jpg')
            siren.image(siren_image, use_column_width=True)

            file_path = 'samples/siren.wav'
            siren.pyplot(draw_spectrogram(file_path))
            st.audio(file_path)

        with street:
            # Street music
            st.subheader('Música callejera')
            music_image = Image.open('Resources/street_music.jpg')
            street.image(music_image, use_column_width=True)

            file_path = 'samples/street.wav'
            street.pyplot(draw_spectrogram(file_path))
            st.audio(file_path)

        st.header('Primeras aproximaciones')
        vgg, cnn_specto = st.columns(2)
        with vgg:
            vgg.subheader('VGG')
            imagevgg = Image.open('Resources/vgg.png')
            vgg.image(imagevgg, use_column_width=True)
        with cnn_specto:
            cnn_specto.subheader('CNN usando espectogramas')
            imagecnn_specto = Image.open('Resources/cnn_specto.png')
            cnn_specto.image(imagecnn_specto, use_column_width=True)

        st.header('Modelos con mejor rendimiento')
        mlp, cnn = st.columns(2)
        with mlp:
            mlp.subheader('MLP')
            imagemlp = Image.open('Resources/MLP.png')
            mlp.image(imagemlp, use_column_width=True)
        with cnn:
            cnn.subheader('CNN')
            imagecnn = Image.open('Resources/SimpleCNN.png')
            cnn.image(imagecnn, use_column_width=True)

    else:
        st.header(user_file.name)
        st.audio(user_file)

        st.subheader('Extracción de características')
        # st.write(extract_features(user_file))

        st.subheader('Predicción')
        clase, proba = predict(user_file, 'cnn')
        details, img = st.columns((1, 2))
        with details:
            details.subheader(f'Clase: {clase}')
            details.dataframe(proba.sort_values(by=['Prob'], ascending=False))

        with img:
            pred_img = Image.open(f'Resources/{clase}_2.jpg')
            img.image(pred_img, use_column_width=True)



if __name__ == '__main__':
    st.set_page_config(
        page_title='Audio Recognition',
        page_icon=':music:',
        layout='wide',
        initial_sidebar_state='collapsed',
        menu_items={
            'Get Help': 'https://www.extremelycoolapp.com/help',
            'Report a bug': "https://www.extremelycoolapp.com/bug"
        }
    )
    main()
