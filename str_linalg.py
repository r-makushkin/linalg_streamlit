import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

st.title("Применение SVD к черно-белому изображению")
image_url = st.text_input("Введите URL изображения и нажмите Enter:")

if image_url:
    try:
        image = io.imread(image_url, as_gray=True)


        U, sing_values, V = np.linalg.svd(image)
        sigma = np.zeros(shape=image.shape)
        np.fill_diagonal(sigma, sing_values)

        top_k = st.sidebar.slider("Выберите значение", 0, image.shape[1], int(image.shape[1]/2))

        trunc_U = U[:, :top_k]
        trunc_sigma = sigma[:top_k, :top_k]
        trunc_V = V[:top_k, :]

        trunc_image = trunc_U@trunc_sigma@trunc_V

        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title("Ваше изображение")
        axes[1].imshow(trunc_image, cmap='gray')
        axes[1].set_title("Изображение после SVD")
        st.pyplot(fig)
    
    except Exception as e:
        st.error(f"Произошла ошибка при загрузке изображения: {e}")

#streamlit run str_linalg.py