import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image

# 페이지 설정
st.set_page_config(page_title="신선도 예측 시스템", layout="wide")

# 제목
st.title("🍎 과일/채소 신선도 예측 시스템")

# 사이드바
st.sidebar.header("이미지 업로드")


def preprocess_image(img, target_size=(240, 240)):
    """
    이미지를 모델 입력 크기에 맞게 전처리
    """
    img = img.resize(target_size)  # 모델 입력 크기로 리사이즈
    img_array = np.array(img)  # 이미지를 NumPy 배열로 변환
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가 -> (1, target_size[0], target_size[1], 3)
    img_array = img_array.astype('float32') / 255.0  # [0, 1] 범위로 정규화
    return img_array


def load_prediction_model(model_path):
    """
    모델 로드 함수
    """
    try:
        model = load_model(model_path)
        st.sidebar.success("모델 로드 완료")
        return model
    except Exception as e:
        st.sidebar.error(f"모델 로드 실패: {e}")
        return None


def predict_freshness(model, img_array, threshold=0.5):
    """
    신선도 예측 함수
    """
    prediction = model.predict(img_array)
    st.subheader("예측 결과")
    st.write(f"모델 예측값: {prediction}")

    predicted_value = prediction[0][0]  # 첫 번째 결과 값
    if predicted_value > threshold:
        st.success(f"신선함 🌟 (신선도 점수: {predicted_value:.2f})")
    else:
        st.error(f"신선하지 않음 ⚠️ (신선도 점수: {predicted_value:.2f})")


def main():
    # 학습된 모델 로드
    model_path = "fruit_classifier3.h5"  # 학습된 모델 파일 경로
    model = load_prediction_model(model_path)
    if model is None:
        return

    # 이미지 업로드
    uploaded_image = st.sidebar.file_uploader("신선도를 판단할 이미지를 업로드하세요", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_image is not None:
        # 업로드된 이미지 표시
        img = Image.open(uploaded_image)
        st.image(img, caption="업로드된 이미지", use_column_width=True)
        
        # 이미지 전처리
        img_array = preprocess_image(img)
        
        # 예측
        try:
            predict_freshness(model, img_array)
        except Exception as e:
            st.error(f"예측 실패: {e}")


if __name__ == '__main__':
    main()
