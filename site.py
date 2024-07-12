import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 엑셀 파일 경로 설정 (실제 파일 경로로 변경하세요)
EXCEL_FILE_PATH = '주히.xlsx'

# 엑셀 파일 로드 및 데이터 전처리
@st.cache_data
def load_data(file_path):
    df = pd.read_excel(file_path)
    df['Q'] = df['Q'].fillna('')
    df['A'] = df['A'].fillna('')
    return df

df = load_data(EXCEL_FILE_PATH)

# TF-IDF 벡터화 도구 학습
@st.cache_resource
def train_vectorizer(data):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(data['Q'].tolist())
    return vectorizer, vectors

question_vectorizer, question_vector = train_vectorizer(df)

# 유사 질문 찾기 함수
def get_most_similar_question(user_question, threshold):
    new_sen_vector = question_vectorizer.transform([user_question])
    simil_score = cosine_similarity(new_sen_vector, question_vector)
    if simil_score.max() < threshold:
        return None, "유사한 질문을 찾을 수 없습니다."
    else:
        max_index = simil_score.argmax()
        most_similar_question = df['Q'].tolist()[max_index]
        most_similar_answer = df['A'].tolist()[max_index]
        return most_similar_question, most_similar_answer

# Streamlit 앱 설정
st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Nanum+Gothic:wght@400;700&display=swap');
       body {{
            background-color: #ffffff; /* 흰색 배경색 */
            font-family: 'Nanum Gothic', sans-serif;
            color: #1E90FF; /* 파란색 텍스트 */
        }}
        .stButton button {{
            margin: 10px 0;
            width: 100%;
            border-radius: 10px;
            border: 1px solid #1E90FF; /* 파란색 테두리 */
            padding: 15px 30px;
            color: #1E90FF; /* 파란색 텍스트 */
            background-color: #ffffff;
            font-size: 1.2em;
            font-weight: 600;
            transition: all 0.3s ease-in-out;
        }}
        .stButton button:hover {{
            background-color: #1E90FF; /* 파란색 배경색 */
            color: #ffffff;
            cursor: pointer;
        }}
        h1 {{
            color: #1E90FF; /* 파란색 */
            text-align: center;
            font-weight: 700;
            margin-bottom: 0.5em;
        }}
        .header {{
            font-size: 2em;
            font-weight: bold;
            color: #1E90FF; /* 파란색 */
            text-align: center;
            margin-bottom: 1em;
        }}
        .section {{
            background-color: rgba(255, 255, 255, 0.8);
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            font-size: 1.2em;
            color: #333333;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            animation: fadeIn 0.5s ease-in-out;
        }}
        .small-button {{
            margin: 10px 5px;
            width: auto;
            border-radius: 5px;
            border: 1px solid #1E90FF; /* 파란색 테두리 */
            padding: 8px 15px;
            color: #1E90FF; /* 파란색 텍스트 */
            background-color: #ffffff;
            font-size: 0.8em;
            font-weight: 600;
            transition: all 0.3s ease-in-out;
        }}
        .small-button:hover {{
            background-color: #1E90FF; /* 파란색 배경색 */
            color: #ffffff;
            cursor: pointer;
        }}
       .stSlider > div > div > div > div {{
            position: relative;
        }}
        <style>
    /* 슬라이더 핸들 스타일 */
    .stSlider > div > div > div > div > div {{
        width: 30px;  /* 핸들의 너비 */
        height: 30px; /* 핸들의 높이 */
        background: #1E90FF !important; /* 핸들의 배경색 */
        clip-path: polygon(
            50% 0%, 
            61% 35%, 
            98% 35%, 
            68% 57%, 
            79% 91%, 
            50% 70%, 
            21% 91%, 
            32% 57%, 
            2% 35%, 
            39% 35%
        ); /* 별 모양 */
        border-radius: 50%; /* 핸들 모서리 둥글게 처리 */
        border: 2px solid #1E90FF; /* 핸들의 테두리 */
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #ffffff;
        font-size: 14px; /* 숫자 크기 */
        font-weight: bold;
    }}

    /* 슬라이더 핸들 안에 숫자 표시 */
    .stSlider > div > div > div > div > div::after {{
        content: attr(data-value);
        display: block;
        font-size: 14px;
        color: #ffffff;
    }}
        
    </style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'active_button' not in st.session_state:
    st.session_state.active_button = None

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = [{"role": "system", "content": "You are a helpful assistant."}]

st.title("^주희의_모든것^")

# 사이드바에 버튼 배열
st.sidebar.header("버튼을 클릭해보세요!")

large_buttons = {
    "나의 아바타": "did.mp4",  # 이 경로를 실제 아바타 비디오 파일로 변경하세요.
    "나를 표현한 음악": "내 마음의 맛.mp3"  # 이 경로를 실제 음악 파일로 변경하세요.
}

small_buttons = {
    "나의 장점": "친구들을 뒤에서 세심하게 도웁니다. 또한 소외되는 친구 없이 모든 친구들이 활동에 적극적으로 참여할 수 있도록 잘 이끌어나갑니다.",
    "희망 진로": "프로그래머가 되고 싶습니다.",
    "오늘의 추천곡": "베이비몬스터 - forever",
    "싫어하는 것": "최선을 다하지 않고 무임승차 하려고 하는 것.",
    "자기 소개": "안녕하세요! 저는 서산중앙고 3학년 6반 김주희 입니다. 반가워요~",
    "진로 준비": "저는 컴퓨터공학과에 진학하기 위해 학교에서 주최하여 열리는 프로그래밍 활동에 모두 참여하고 있습니다. 또한 동아리에서 부장을 맡고 있습니다.",
    "취미 활동": "저는 게임하는 것을 무척 좋아합니다. 대신 총게임은 별로 안 좋아해요. 그리고 노이즈 켄슬링 키고 정적 속에서 큰소리로 노래 듣는 것을 좋아합니다",
    "진로 활동": "저는 학교 친구들과 예산에서 복싱로봇 부스를 운영하여 좋은 반응을 받은 적이 있습니다.",
    "인스타 아이디": "3ch._.1"
}

for button, content in large_buttons.items():
    if st.sidebar.button(button, key=button):
        st.session_state.active_button = button if st.session_state.active_button != button else None

if st.session_state.active_button == "나의 아바타":
    st.video("did.mp4", format="video/mp4", start_time=0)

if st.session_state.active_button == "나를 표현한 음악":
    st.audio("내 마음의 맛.mp3", format="audio/mp3")


for button, content in small_buttons.items():
    if st.sidebar.button(button, key=button):
        st.session_state.active_button = button if st.session_state.active_button != button else None
    if st.session_state.active_button == button:
        st.markdown(f"<div class='section'>{content}</div>", unsafe_allow_html=True)


# 유사도 임계값 슬라이더 추가
threshold = st.slider("유사도 임계값", 0.0, 1.0, 0.43)

# 사용자 입력을 받는 입력 창
user_input = st.text_input("질문을 입력하세요:")

# 검색 버튼
if st.button("검색", key="search", help="small"):
    if user_input:
        # 유사 질문 찾기
        similar_question, answer = get_most_similar_question(user_input, threshold)
        
        if similar_question:
            st.session_state.conversation_history.append({"role": "assistant", "content": f"유사한 질문: {similar_question}"})
            st.session_state.conversation_history.append({"role": "assistant", "content": answer})
            st.write(f"**유사한 질문:** {similar_question}")
            st.write(f"**답변:** {answer}")
        else:
            st.write("유사한 질문을 찾을 수 없습니다.")

# 이전 대화 보기 버튼
if st.button("이전 대화 보기", key="view_history", help="small"):
    st.write("### 대화 기록")
    for msg in st.session_state.conversation_history:
        role = "You" if msg["role"] == "user" else "Assistant"
        st.write(f"**{role}:** {msg['content']}")

# 새 검색 시작 버튼
if st.button("새 검색 시작", key="new_search", help="small"):
    st.session_state.conversation_history = [{"role": "system", "content": "You are a helpful assistant."}]
    st.experimental_rerun()  # 페이지를 새로고침하여 대화 기록을 초기화

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
