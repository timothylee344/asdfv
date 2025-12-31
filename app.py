import streamlit as st
import cv2
import numpy as np
from PIL import Image
import openai
from io import BytesIO

# 페이지 설정
st.set_page_config(
    page_title="JEJUNUA 피부 진단",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS 스타일
st.markdown("""
    <style>
    /* 전체 배경 및 테마 */
    .stApp {
        background: linear-gradient(135deg, #001a14 0%, #004d40 100%);
    }
    
    /* 메인 컨테이너 */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* 제목 스타일 */
    h1 {
        color: #FFD700;
        font-family: 'Georgia', serif;
        text-align: center;
        font-weight: 600;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        margin-bottom: 0.5rem;
    }
    
    h2, h3 {
        color: #FFD700;
        font-family: 'Georgia', serif;
    }
    
    /* 버튼 스타일 */
    .stButton > button {
        background: linear-gradient(135deg, #004d40 0%, #00695c 100%);
        color: #FFD700;
        border: 2px solid #FFD700;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #00695c 0%, #00897b 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(255,215,0,0.3);
    }
    
    /* 텍스트 입력 스타일 */
    .stTextInput > div > div > input {
        background-color: rgba(0, 77, 64, 0.5);
        color: #FFD700;
        border: 1px solid #FFD700;
        border-radius: 10px;
    }
    
    /* 사이드바 스타일 */
    .css-1d391kg {
        background-color: rgba(0, 26, 20, 0.95);
    }
    
    /* 메트릭 스타일 */
    [data-testid="stMetricValue"] {
        color: #FFD700;
        font-size: 2.5rem;
        font-weight: bold;
    }
    
    [data-testid="stMetricLabel"] {
        color: #B2DFDB;
    }
    
    /* 정보 박스 스타일 */
    .stInfo {
        background-color: rgba(0, 77, 64, 0.3);
        border-left: 4px solid #FFD700;
        border-radius: 10px;
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

def analyze_redness(image):
    """
    OpenCV를 사용하여 이미지에서 붉은기 영역을 분석합니다.
    """
    # PIL Image를 numpy array로 변환
    img_array = np.array(image)
    
    # 채널 수 확인 및 변환 (RGBA -> RGB)
    if len(img_array.shape) == 3:
        if img_array.shape[2] == 4:  # RGBA (4채널)
            # 알파 채널 제거하여 RGB로 변환
            img_array = img_array[:, :, :3]
        elif img_array.shape[2] == 1:  # Grayscale
            # 그레이스케일을 RGB로 변환
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] != 3:  # 기타 채널 수
            # 처음 3개 채널만 사용
            img_array = img_array[:, :, :3]
    elif len(img_array.shape) == 2:  # 2D 배열 (그레이스케일)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    # 최종적으로 3채널 RGB인지 확인
    if len(img_array.shape) != 3 or img_array.shape[2] != 3:
        raise ValueError(f"이미지 채널 변환 실패: 현재 shape = {img_array.shape}")
    
    # RGB를 BGR로 변환 (OpenCV는 BGR 사용)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # HSV 색공간으로 변환
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # 붉은색 범위 정의 (HSV)
    # HSV에서 빨간색은 0도와 180도 근처에 있음
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # 붉은색 영역 마스크 생성
    mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # 노이즈 제거
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    
    # 붉은기 면적 계산
    total_pixels = img_array.shape[0] * img_array.shape[1]
    red_pixels = np.sum(red_mask > 0)
    redness_percentage = (red_pixels / total_pixels) * 100
    
    # 히트맵 생성
    # 붉은 영역을 빨간색으로 강조 (3채널 RGB 보장)
    heatmap = img_array.copy()
    if heatmap.shape[2] == 3:
        heatmap[red_mask > 0] = [255, 0, 0]  # 빨간색으로 표시 (RGB)
    else:
        # 안전장치: 채널 수가 맞지 않으면 RGB로 변환 후 처리
        heatmap = heatmap[:, :, :3] if heatmap.shape[2] > 3 else heatmap
        heatmap[red_mask > 0] = [255, 0, 0]
    
    # 원본 이미지와 히트맵을 블렌딩 (투명도 적용)
    overlay = img_array.copy()
    if overlay.shape[2] == 3:
        overlay[red_mask > 0] = [255, 0, 0]
    else:
        overlay = overlay[:, :, :3] if overlay.shape[2] > 3 else overlay
        overlay[red_mask > 0] = [255, 0, 0]
    
    # 블렌딩 전 채널 수 최종 확인
    if img_array.shape[2] != 3 or overlay.shape[2] != 3:
        img_array = img_array[:, :, :3] if img_array.shape[2] > 3 else img_array
        overlay = overlay[:, :, :3] if overlay.shape[2] > 3 else overlay
    
    heatmap_blended = cv2.addWeighted(img_array, 0.7, overlay, 0.3, 0)
    
    return heatmap_blended, redness_percentage, red_mask

def get_skin_advice(redness_percentage, api_key):
    """
    OpenAI API를 사용하여 피부 관리 조언을 받습니다.
    """
    try:
        client = openai.OpenAI(api_key=api_key)
        
        prompt = f"""당신은 친절하고 전문적인 에스테틱 전문가입니다. 
고객의 피부 붉은기 수치가 {redness_percentage:.2f}%로 분석되었습니다.
이 수치를 바탕으로 친절하고 따뜻한 말투로 피부 관리 조언을 해주세요.
조언은 2-3문단으로 간결하게 작성하고, 실용적인 팁을 포함해주세요.
말투는 전문적이면서도 친근하게, 고객을 배려하는 톤으로 작성해주세요."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 프리미엄 화장품 브랜드 JEJUNUA의 전문 에스테틱 전문가입니다. 고객에게 친절하고 전문적인 피부 관리 조언을 제공합니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"조언을 생성하는 중 오류가 발생했습니다: {str(e)}"

# 메인 앱
def main():
    # 제목
    st.markdown("<h1>✨ JEJUNUA 프리미엄 피부 진단 ✨</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #B2DFDB; font-size: 1.1rem; margin-bottom: 2rem;'>당신의 피부를 정밀하게 분석하고 맞춤형 조언을 제공합니다</p>", unsafe_allow_html=True)
    
    # 사이드바 설정
    with st.sidebar:
        st.markdown("<h2 style='color: #FFD700;'>⚙️ 설정</h2>", unsafe_allow_html=True)
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="OpenAI API 키를 입력하세요. https://platform.openai.com/api-keys 에서 발급받을 수 있습니다.",
            placeholder="sk-..."
        )
        
        st.markdown("---")
        st.markdown("""
        <div style='color: #B2DFDB; font-size: 0.9rem;'>
        <h3 style='color: #FFD700;'>📱 사용 방법</h3>
        <ol style='padding-left: 1.2rem;'>
            <li>OpenAI API Key를 입력하세요</li>
            <li>사진을 업로드하거나 카메라로 촬영하세요</li>
            <li>분석 결과와 조언을 확인하세요</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    # 사진 업로드
    st.markdown("<h2 style='color: #FFD700; margin-top: 2rem;'>📸 사진 업로드</h2>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "피부 사진을 업로드하거나 카메라로 촬영하세요",
        type=['png', 'jpg', 'jpeg'],
        help="얼굴이나 피부 부위의 사진을 업로드해주세요"
    )
    
    if uploaded_file is not None:
        # 이미지 로드 및 RGBA -> RGB 변환
        image = Image.open(uploaded_file)
        # RGBA 모드인 경우 RGB로 변환
        if image.mode == 'RGBA':
            # 흰색 배경에 합성하여 RGB로 변환
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[3])  # 알파 채널을 마스크로 사용
            image = rgb_image
        elif image.mode != 'RGB':
            # 기타 모드(P, L 등)도 RGB로 변환
            image = image.convert('RGB')
        
        # 이미지 표시
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3 style='color: #FFD700;'>📷 원본 사진</h3>", unsafe_allow_html=True)
            st.image(image, use_container_width=True)
        
        # 분석 버튼
        if st.button("🔍 피부 분석 시작", use_container_width=True):
            if not api_key:
                st.error("⚠️ OpenAI API Key를 먼저 입력해주세요.")
            else:
                with st.spinner("피부를 분석하는 중입니다..."):
                    # 붉은기 분석
                    heatmap, redness_perc, mask = analyze_redness(image)
                    
                    with col2:
                        st.markdown("<h3 style='color: #FFD700;'>🔥 붉은기 분석 결과</h3>", unsafe_allow_html=True)
                        st.image(heatmap, use_container_width=True)
                    
                    # 결과 표시
                    st.markdown("---")
                    st.markdown("<h2 style='color: #FFD700;'>📊 분석 결과</h2>", unsafe_allow_html=True)
                    
                    # 메트릭 표시
                    col3, col4, col5 = st.columns(3)
                    with col3:
                        st.metric(
                            label="붉은기 면적",
                            value=f"{redness_perc:.2f}%",
                            help="전체 피부 면적 대비 붉은기 비율"
                        )
                    
                    with col4:
                        if redness_perc < 5:
                            status = "양호"
                            delta_color = "normal"
                        elif redness_perc < 15:
                            status = "주의"
                            delta_color = "off"
                        else:
                            status = "관리 필요"
                            delta_color = "inverse"
                        st.metric(
                            label="피부 상태",
                            value=status
                        )
                    
                    with col5:
                        st.metric(
                            label="분석 완료",
                            value="✓",
                            help="분석이 완료되었습니다"
                        )
                    
                    # OpenAI 조언
                    st.markdown("---")
                    st.markdown("<h2 style='color: #FFD700;'>💡 전문가 조언</h2>", unsafe_allow_html=True)
                    
                    with st.spinner("전문가 조언을 생성하는 중입니다..."):
                        advice = get_skin_advice(redness_perc, api_key)
                        
                        st.info(f"💬 {advice}")
                    
                    # 추가 정보
                    st.markdown("---")
                    st.markdown("""
                    <div style='background-color: rgba(0, 77, 64, 0.3); padding: 1.5rem; border-radius: 10px; border-left: 4px solid #FFD700;'>
                    <h3 style='color: #FFD700; margin-top: 0;'>ℹ️ 분석 정보</h3>
                    <p style='color: #B2DFDB;'>
                    • 붉은기 분석은 HSV 색공간을 기반으로 수행됩니다.<br>
                    • 결과는 참고용이며, 정확한 진단은 전문의와 상담하시기 바랍니다.<br>
                    • 정확한 분석을 위해 조명이 균일한 환경에서 촬영해주세요.
                    </p>
                    </div>
                    """, unsafe_allow_html=True)
    
    else:
        st.info("👆 위에서 사진을 업로드하거나 카메라로 촬영해주세요.")

if __name__ == "__main__":
    main()

