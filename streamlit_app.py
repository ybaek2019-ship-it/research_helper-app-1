import streamlit as st
from io import BytesIO
from pypdf import PdfReader
import re
import unicodedata
from openai import OpenAI
import json
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import csv
import io
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 상수 정의
MAX_FILE_SIZE_MB = 30
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# API 키 관리
CONFIG_DIR = Path(__file__).parent / "config"
CONFIG_FILE = CONFIG_DIR / "api_keys.json"

def load_api_key():
    """API 키를 로드합니다."""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key
    
    try:
        if hasattr(st, 'secrets') and 'default' in st.secrets and 'openai_api_key' in st.secrets['default']:
            return st.secrets['default']['openai_api_key']
    except:
        pass
    
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                return config.get('openai_api_key')
    except:
        pass
    
    return None

@st.cache_resource
def get_openai_client():
    """OpenAI 클라이언트를 초기화합니다."""
    api_key = load_api_key()
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

# ==================== GPT 기반 분석 함수 ====================

def gpt_analyze_all(text, max_words=3500):
    """GPT를 사용하여 논문을 종합적으로 분석합니다."""
    try:
        client = get_openai_client()
        if not client:
            return {"error": "OpenAI API 키가 설정되지 않았습니다."}
        
        words = text.split()
        truncated_text = ' '.join(words[:max_words])
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 학술 논문 분석 전문가입니다. 질적 연구방법론에 특히 정통하며, 한국어로 명확하고 상세한 분석을 제공합니다. **중요: 논문에 명시된 사실과 당신의 추론/해석을 명확히 구분하여 표기하세요.**"},
                {"role": "user", "content": f"""다음 학술 논문을 종합적으로 분석하여 한국어로 답변해주세요:

{truncated_text}

다음 섹션별로 명확하게 구분하여 작성해주세요.
**중요 규칙**: 각 내용 앞에 [사실] 또는 [추론] 태그를 붙여 출처를 명확히 하세요.
- [사실]: 논문에 명시적으로 기술된 내용
- [추론]: AI가 추론하거나 해석한 내용

[핵심요약]
3-5문장으로 논문의 핵심 내용을 요약

[연구목적]
연구의 목적과 배경 설명

[연구방법]
사용된 연구방법론 상세 설명 (참여자, 자료수집, 분석방법 포함)

[주요발견]
핵심 연구 결과 및 발견사항

[이론적기여]
이론적/실천적 함의와 기여

[한계점]
연구의 한계점 및 향후 연구 방향"""}
            ],
            temperature=0.3,
            max_tokens=2500
        )
        
        result = response.choices[0].message.content
        
        # 섹션별로 파싱
        sections = {}
        current_section = None
        current_content = []
        
        for line in result.split('\n'):
            if line.strip().startswith('[') and line.strip().endswith(']'):
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = line.strip()[1:-1]
                current_content = []
            else:
                if current_section and line.strip():
                    current_content.append(line)
        
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections if sections else {"핵심요약": result}
        
    except Exception as e:
        return {"error": f"GPT 분석 실패: {str(e)}"}

def gpt_analyze_structure(text, max_words=3000):
    """GPT를 사용하여 논문 구조를 분석합니다."""
    try:
        client = get_openai_client()
        if not client:
            return {"error": "OpenAI API 키가 설정되지 않았습니다."}
        
        words = text.split()
        truncated_text = ' '.join(words[:max_words])
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 학술 논문의 구조를 분석하는 전문가입니다. IMRaD 구조(서론, 방법, 결과, 논의)를 잘 이해하고 있습니다. **중요: 논문에 명시된 사실과 추론을 구분하여 표기하세요.**"},
                {"role": "user", "content": f"""다음 논문의 구조를 분석하여 각 섹션을 요약해주세요.
**중요**: 각 내용 앞에 [사실] 또는 [추론] 태그를 붙이세요.

{truncated_text}

다음 형식으로 작성해주세요:

[서론_배경]
서론 및 연구 배경 요약 (3-5문장)

[이론적_프레임워크]
이론적 틀 및 선행연구 요약 (3-5문장)

[연구방법]
연구설계, 참여자, 자료수집 방법 상세 설명

[자료분석]
자료 분석 절차 및 기법 설명

[연구결과]
주요 연구 결과 요약

[논의_함의]
논의 및 실천적 함의"""}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        result = response.choices[0].message.content
        
        # 섹션별로 파싱
        sections = {}
        current_section = None
        current_content = []
        
        for line in result.split('\n'):
            if line.strip().startswith('[') and line.strip().endswith(']'):
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = line.strip()[1:-1]
                current_content = []
            else:
                if current_section and line.strip():
                    current_content.append(line)
        
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections if sections else {"error": "구조 분석 실패"}
        
    except Exception as e:
        return {"error": f"구조 분석 실패: {str(e)}"}

def gpt_analyze_keywords_themes(text, max_words=3000):
    """GPT를 사용하여 주제와 키워드를 분석합니다."""
    try:
        client = get_openai_client()
        if not client:
            return {"error": "OpenAI API 키가 설정되지 않았습니다."}
        
        words = text.split()
        truncated_text = ' '.join(words[:max_words])
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 학술 논문의 주제와 키워드를 추출하는 전문가입니다. **중요: 논문에 명시된 사실과 추론을 구분하여 표기하세요.**"},
                {"role": "user", "content": f"""다음 논문에서 연구질문, 주요 주제, 키워드를 추출해주세요.
**중요**: 각 항목 앞에 [사실] (논문에 명시됨) 또는 [추론] (AI 추출) 태그를 붙이세요.

{truncated_text}

다음 형식으로 작성해주세요:

[연구질문]
- RQ1: 첫 번째 연구질문
- RQ2: 두 번째 연구질문
- RQ3: 세 번째 연구질문

[연구가설]
- H1: 첫 번째 가설
- H2: 두 번째 가설

[주요주제]
- 주제1: 첫 번째 주요 주제
- 주제2: 두 번째 주요 주제
- 주제3: 세 번째 주요 주제
- 주제4: 네 번째 주요 주제
- 주제5: 다섯 번째 주요 주제

[핵심개념]
개념1, 개념2, 개념3, 개념4, 개념5

[중요키워드]
키워드1, 키워드2, 키워드3, 키워드4, 키워드5, 키워드6, 키워드7, 키워드8, 키워드9, 키워드10

[학술용어]
용어1, 용어2, 용어3, 용어4, 용어5, 용어6, 용어7

주의: 연구질문이나 가설이 명시되지 않은 경우, 논문의 목적을 기반으로 추론하여 작성해주세요."""}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        
        result = response.choices[0].message.content
        
        # 섹션별로 파싱
        sections = {}
        current_section = None
        current_content = []
        
        for line in result.split('\n'):
            if line.strip().startswith('[') and line.strip().endswith(']'):
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = line.strip()[1:-1]
                current_content = []
            else:
                if current_section and line.strip():
                    current_content.append(line)
        
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections if sections else {"error": "주제 분석 실패"}
        
    except Exception as e:
        return {"error": f"주제 분석 실패: {str(e)}"}

def gpt_analyze_references(text):
    """GPT를 사용하여 참고문헌을 분석합니다."""
    try:
        client = get_openai_client()
        if not client:
            return {"error": "OpenAI API 키가 설정되지 않았습니다."}
        
        # References 섹션 찾기 - 더 넓은 범위로 검색
        ref_section = ""
        patterns = [
            r'References\s*\n(.*?)(?=\n\n[A-Z][a-z]+|\Z)',
            r'REFERENCES\s*\n(.*?)(?=\n\n[A-Z][a-z]+|\Z)',
            r'Bibliography\s*\n(.*?)(?=\n\n[A-Z][a-z]+|\Z)',
            r'참고문헌\s*\n(.*?)(?=\n\n|\Z)',
            r'References\s+(.*)',
            r'REFERENCES\s+(.*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                ref_section = match.group(1)[:8000]  # 더 많은 텍스트 포함
                break
        
        # 참고문헌이 없으면 텍스트 끝부분 사용
        if not ref_section or len(ref_section) < 200:
            # 텍스트의 마지막 20% 사용
            last_part = text[int(len(text) * 0.8):]
            if len(last_part) > 500:
                ref_section = last_part[:8000]
        
        if not ref_section or len(ref_section) < 200:
            return {"error": "참고문헌 섹션을 찾을 수 없습니다. 논문에 참고문헌이 포함되어 있는지 확인해주세요."}
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 학술 논문의 참고문헌을 분석하는 전문가입니다. 서지정보를 정확히 추출하고 대학원생에게 유용한 인사이트를 제공합니다. **중요: 사실과 추론을 구분하여 표기하세요.**"},
                {"role": "user", "content": f"""다음 참고문헌 목록을 분석하여 대학원생이 문헌 조사에 활용할 수 있도록 상세히 정리해주세요.
**중요**: [통계요약]과 [핵심문헌]은 [사실], [시사점]은 [추론]으로 명확히 구분하세요.

{ref_section}

다음 형식으로 작성해주세요:

[통계요약]
• 총 참고문헌: XX개
• 연도 범위: XXXX-XXXX년
• 최근 5년 이내: XX개 (XX%)
• 평균 저자수: X.X명

[핵심문헌]
각 문헌을 다음 형식으로 나열 (최대 8개):
• 저자(연도). 제목. 저널/출판사.
  → [사실] 이 논문에서 X회 인용됨 (또는 참고문헌 목록에 포함된 사실)
  → [추론] 이 분야의 이론적 기초를 제공/연구방법론을 제시/핵심 실증연구 등의 추천 사유

[주요저널]
• Journal Name 1 (XX회 인용)
• Journal Name 2 (XX회 인용)
• Journal Name 3 (XX회 인용)

[영향력있는연구자]
• 연구자1 (XX회 인용) - 주요 연구 주제
• 연구자2 (XX회 인용) - 주요 연구 주제
• 연구자3 (XX회 인용) - 주요 연구 주제

[출판물유형]
• 저널논문: XX개
• 단행본/저서: XX개
• 학술대회: XX개
• 학위논문: XX개
• 기타: XX개

[시사점]
이 참고문헌 목록이 보여주는 연구 흐름, 주요 이론적 기반, 또는 연구방법론적 특징을 2-3문장으로 요약"""}
            ],
            temperature=0.2,
            max_tokens=2000
        )
        
        result = response.choices[0].message.content
        
        # 섹션별로 파싱
        sections = {}
        current_section = None
        current_content = []
        
        for line in result.split('\n'):
            if line.strip().startswith('[') and line.strip().endswith(']'):
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = line.strip()[1:-1]
                current_content = []
            else:
                if current_section and line.strip():
                    current_content.append(line)
        
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections if sections else {"error": "참고문헌 분석 실패"}
        
    except Exception as e:
        return {"error": f"참고문헌 분석 실패: {str(e)}"}

# ==================== 논문 비교 분석 ====================

def gpt_compare_papers(papers_data, max_words_per_paper=2000):
    """GPT를 사용하여 여러 논문을 심층 비교합니다."""
    try:
        client = get_openai_client()
        if not client:
            return {"error": "OpenAI API 키가 설정되지 않았습니다."}
        
        if len(papers_data) < 2:
            return {"error": "비교를 위해서는 최소 2개의 논문이 필요합니다."}
        
        # 각 논문의 주요 정보 추출
        papers_summary = []
        for name, data in papers_data.items():
            text = data.get('text', '')
            words = text.split()[:max_words_per_paper]
            truncated = ' '.join(words)
            
            # 기존 분석 결과 활용
            main_analysis = data.get('main_analysis', {})
            keywords = data.get('keywords_themes', {})
            
            summary = f"""
논문명: {name}
연구목적: {main_analysis.get('연구목적', 'N/A')[:200]}
연구방법: {main_analysis.get('연구방법', 'N/A')[:200]}
주요발견: {main_analysis.get('주요발견', 'N/A')[:200]}
연구질문: {keywords.get('연구질문', 'N/A')[:200]}
"""
            papers_summary.append(summary)
        
        combined_summary = "\n\n---\n\n".join(papers_summary)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 학술 논문 비교분석 전문가입니다. 대학원생이 문헌고찰과 연구 설계에 활용할 수 있도록 실질적인 인사이트를 제공합니다."},
                {"role": "user", "content": f"""다음 논문들을 비교 분석하여 대학원생의 연구에 도움이 되도록 답변해주세요:

{combined_summary}

다음 형식으로 작성해주세요:

[연구공백]
두 논문이 공통으로 다룬 주제와 각각이 다루지 않은 영역을 분석하여, 새로운 연구 기회를 3-5개 제시하세요.

[방법론비교]
각 논문의 연구방법(정량/정성/혼합, 표본, 데이터수집)을 비교하고, 각 방법론의 장단점을 설명하세요.

[이론적차이]
각 논문이 사용한 이론적 프레임워크를 비교하고, 어떤 상황에 어떤 이론이 적합한지 설명하세요.

[주요차별점]
두 논문의 핵심적인 차이점 3가지를 명확히 제시하세요.

[연구제안]
이 두 논문을 바탕으로 새로운 연구를 설계한다면 어떤 방향이 좋을지 구체적으로 제안하세요."""}
            ],
            temperature=0.4,
            max_tokens=2000
        )
        
        result = response.choices[0].message.content
        
        # 섹션별로 파싱
        sections = {}
        current_section = None
        current_content = []
        
        for line in result.split('\n'):
            if line.strip().startswith('[') and line.strip().endswith(']'):
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = line.strip()[1:-1]
                current_content = []
            else:
                if current_section and line.strip():
                    current_content.append(line)
        
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections if sections else {"분석결과": result}
        
    except Exception as e:
        return {"error": f"논문 비교 실패: {str(e)}"}

# ==================== 텍스트 전처리 ====================
def clean_text(text):
    """텍스트를 정제하고 정규화합니다."""
    text = unicodedata.normalize('NFKD', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    return text.strip()

# ==================== PDF 로드 ====================
def load_pdf_from_upload(uploaded_file):
    """업로드된 PDF 파일을 로드합니다."""
    try:
        file_size = uploaded_file.size
        file_size_mb = file_size / 1024 / 1024
        
        if file_size == 0:
            return None, "❌ 업로드된 파일이 비어있습니다."
        
        if file_size > MAX_FILE_SIZE_BYTES:
            return None, f"❌ 파일 크기가 {MAX_FILE_SIZE_MB}MB를 초과합니다.\n현재 파일: {file_size_mb:.2f}MB\n\n💡 PDF 압축을 권장합니다: smallpdf.com"
        
        if file_size_mb > 20:
            st.warning(f"⚠️ 파일 크기가 {file_size_mb:.2f}MB입니다. 처리 시간이 오래 걸릴 수 있습니다.")
        
        uploaded_file.seek(0)
        content = BytesIO(uploaded_file.read())
        content.seek(0)
        
        header = content.read(4)
        content.seek(0)
        if header != b'%PDF':
            return None, "❌ 유효한 PDF 파일이 아닙니다."
        
        return content, None
    except Exception as e:
        return None, f"❌ 파일을 로드할 수 없습니다: {str(e)}"

# ==================== 텍스트 추출 ====================
def extract_text(pdf_file):
    """PDF에서 텍스트를 추출하고 메타데이터를 수집합니다."""
    try:
        pdf_file.seek(0)
        reader = PdfReader(pdf_file)
        
        if len(reader.pages) == 0:
            return None, None, "❌ PDF 파일에 페이지가 없습니다."
        
        metadata = {
            'pages': len(reader.pages),
            'title': None,
            'author': None,
            'subject': None,
            'creator': None
        }
        
        if reader.metadata:
            metadata['title'] = reader.metadata.get('/Title', None)
            metadata['author'] = reader.metadata.get('/Author', None)
            metadata['subject'] = reader.metadata.get('/Subject', None)
            metadata['creator'] = reader.metadata.get('/Creator', None)
        
        text = ""
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
            except Exception as e:
                continue
        
        if not text or len(text.strip()) < 100:
            return None, None, "❌ PDF에서 텍스트를 추출할 수 없습니다. 이미지 기반 PDF이거나 보호된 파일일 수 있습니다."
        
        text = clean_text(text)
        return text, metadata, None
        
    except Exception as e:
        error_msg = str(e)
        if "empty file" in error_msg.lower():
            return None, None, "❌ 빈 파일이거나 손상된 PDF입니다."
        elif "encrypted" in error_msg.lower():
            return None, None, "❌ 암호화된 PDF입니다."
        else:
            return None, None, f"❌ PDF 텍스트 추출 실패: {error_msg}"

# ==================== Streamlit UI ====================
def main():
    st.set_page_config(
        page_title="AI 학술 논문 분석 도구",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1f77b4;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #666;
            margin-bottom: 2rem;
        }
        .section-header {
            color: #1f77b4;
            border-bottom: 2px solid #1f77b4;
            padding-bottom: 0.5rem;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header">📚 용민쌤의 학술 논문 분석 도구</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">대학원생을 위한 지능형 학술논문 분석 시스템</div>', unsafe_allow_html=True)
    
    # 세션 상태 초기화
    if 'papers' not in st.session_state:
        st.session_state.papers = {}
    
    # 사이드바
    with st.sidebar:
        st.header("📤 PDF 업로드")
        
        with st.expander("ℹ️ 사용 가이드", expanded=False):
            st.markdown("""
            **📊 분석 기능:**
            - 종합 분석 (요약, 주제, 키워드)
            - 구조 분석 (서론, 방법, 결과, 논의)
            - 참고문헌 심층 분석
            - 키워드 개념도 시각화
            - 인용 네트워크 시각화
            
            **📁 파일 크기:**
            - 권장: 10MB 이하
            - 최대: 30MB
            - 20MB 이상: 압축 권장
            """)
        
        st.markdown(f"**📊 파일 크기 제한: {MAX_FILE_SIZE_MB}MB**")
        st.caption("⚠️ 20MB 이상 파일은 PDF 압축을 권장합니다. (smallpdf.com)")
        
        uploaded_file = st.file_uploader(
            "PDF 파일을 선택하세요",
            type=['pdf'],
            help=f"학술 논문 PDF 파일 (최대: {MAX_FILE_SIZE_MB}MB)"
        )
        
        paper_name = st.text_input(
            "논문 제목 (선택사항)",
            placeholder="예: Smith et al. (2023)",
            help="비워두면 파일명이 사용됩니다"
        )
        
        analyze_button = st.button("🔍 분석 시작", type="primary", use_container_width=True)
        
        if analyze_button:
            if not uploaded_file:
                st.error("❌ PDF 파일을 먼저 업로드해주세요.")
            elif not load_api_key():
                st.error("❌ OpenAI API 키가 필요합니다.")
            else:
                with st.spinner("📄 PDF 처리 중..."):
                    pdf_content, error = load_pdf_from_upload(uploaded_file)
                    
                    if error:
                        st.error(error)
                    else:
                        with st.spinner("📝 텍스트 추출 중..."):
                            text, metadata, extract_error = extract_text(pdf_content)
                            
                            if extract_error:
                                st.error(extract_error)
                            else:
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                status_text.text("📊 종합 분석 중...")
                                progress_bar.progress(20)
                                main_analysis = gpt_analyze_all(text)
                                
                                status_text.text("📊 구조 분석 중...")
                                progress_bar.progress(40)
                                structure = gpt_analyze_structure(text)
                                
                                status_text.text("📊 주제&키워드 분석 중...")
                                progress_bar.progress(60)
                                keywords_themes = gpt_analyze_keywords_themes(text)
                                
                                status_text.text("📊 참고문헌 분석 중...")
                                progress_bar.progress(80)
                                references = gpt_analyze_references(text)
                                
                                name = paper_name.strip() if paper_name.strip() else uploaded_file.name.replace('.pdf', '')
                                st.session_state.papers[name] = {
                                    'text': text,
                                    'metadata': metadata,
                                    'main_analysis': main_analysis,
                                    'structure': structure,
                                    'keywords_themes': keywords_themes,
                                    'references': references
                                }
                                
                                progress_bar.progress(100)
                                status_text.text("✅ 분석 완료!")
                                
                                st.success(f"**'{name}'** 분석이 완료되었습니다!")
                                st.balloons()
        
        # 로드된 논문 목록
        if st.session_state.papers:
            st.markdown("---")
            st.subheader("📚 분석된 논문")
            
            for idx, name in enumerate(st.session_state.papers.keys(), 1):
                col1, col2 = st.columns([4, 1])
                with col1:
                    pages = st.session_state.papers[name]['metadata']['pages']
                    st.write(f"**{idx}.** {name}")
                    if pages:
                        st.caption(f"📄 {pages} 페이지")
                with col2:
                    if st.button("🗑️", key=f"del_{name}"):
                        del st.session_state.papers[name]
                        st.rerun()
            
            if len(st.session_state.papers) > 1:
                st.info(f"💡 {len(st.session_state.papers)}개 논문 비교 가능")
    
    # 메인 영역
    if not st.session_state.papers:
        st.info("👈 **시작하기:** 왼쪽 사이드바에서 PDF 파일을 업로드하고 AI 분석을 시작하세요.")
        
        # 활용 목적 및 방법
        st.markdown("---")
        st.markdown("### 📖 이 도구의 활용 목적")
        st.markdown("""
        <div style="background-color: #f0f8ff; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4; margin-bottom: 20px;">
        <p style="font-size: 15px; line-height: 1.8; margin: 0;">
        본 도구는 <b>대학원생의 학술 논문 이해를 돕기 위한</b> 분석 보조 도구입니다.<br>
        논문의 핵심 내용을 빠르게 파악하고, 시각화를 통해 개념 간 관계를 직관적으로 이해할 수 있습니다.
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🎯 주요 기능")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### 📊 논문 분석")
            st.write("• **종합 분석**: 논문 요약 및 핵심 내용")
            st.write("• **구조 분석**: 서론, 방법, 결과, 논의")
            st.write("• **키워드 추출**: 주요 개념 및 연구질문")
            st.write("• **참고문헌**: 핵심 문헌 및 연구 동향")
        with col2:
            st.markdown("#### 📈 시각화")
            st.write("• **키워드 개념도**: 주제-키워드 관계")
            st.write("• **인용 네트워크**: 저자-논문 관계")
            st.write("• **CSV 다운로드**: 분석 결과 내보내기")
        with col3:
            st.markdown("#### ⚠️ 신뢰성 구분")
            st.write("• **[사실]**: 논문에 명시된 내용")
            st.write("• **[추론]**: 분석을 통한 해석")
            st.write("• 정보 출처 구분 표기")
        
        st.markdown("---")
        st.markdown("### 💡 올바른 활용 방법")
        st.markdown("""
        <div style="background-color: #fff8dc; padding: 15px; border-radius: 8px; border-left: 4px solid #FFA500;">
        <p style="margin: 5px 0;"><b>✅ 권장:</b> 논문 초기 이해를 위한 보조 도구로 활용</p>
        <p style="margin: 5px 0;"><b>✅ 권장:</b> 분석 결과를 원문과 대조하여 검증</p>
        <p style="margin: 5px 0;"><b>✅ 권장:</b> 참고문헌 조사 시 핵심 문헌 파악용</p>
        <p style="margin: 5px 0; margin-top: 10px;"><b>⚠️ 주의:</b> 분석 결과를 무비판적으로 인용하지 말 것</p>
        <p style="margin: 5px 0;"><b>⚠️ 주의:</b> 네트워크 시각화는 추정값이므로 원문 확인 필요</p>
        <p style="margin: 5px 0;"><b>⚠️ 주의:</b> 학술 연구는 반드시 원문을 직접 읽고 비판적으로 분석</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<p style="text-align: center; color: #888; font-size: 0.85rem;">대학원 연구 보조 목적</p>', unsafe_allow_html=True)
    
    else:
        # 논문 선택 및 CSV 다운로드 버튼
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_paper = st.selectbox(
                "📖 분석할 논문 선택",
                options=list(st.session_state.papers.keys()),
                key="paper_selector"
            )
        
        data = st.session_state.papers[selected_paper]
        meta = data['metadata']
        
        # CSV 데이터 생성 함수
        def generate_csv_data():
            csv_rows = []
            
            # 메타데이터
            csv_rows.append(['=== 문서 정보 ===', ''])
            csv_rows.append(['논문명', selected_paper])
            csv_rows.append(['제목', meta.get('title', '')])
            csv_rows.append(['저자', meta.get('author', '')])
            csv_rows.append(['페이지 수', str(meta.get('pages', ''))])
            csv_rows.append(['작성 도구', meta.get('creator', '')])
            csv_rows.append(['', ''])
            
            # 종합 분석
            analysis = data.get('main_analysis', {})
            if analysis and 'error' not in analysis:
                csv_rows.append(['=== 종합 분석 ===', ''])
                for key, value in analysis.items():
                    csv_rows.append([key, value.replace('\n', ' ') if value else ''])
                csv_rows.append(['', ''])
            
            # 구조 분석
            structure = data.get('structure', {})
            if structure and 'error' not in structure:
                csv_rows.append(['=== 구조 분석 ===', ''])
                for key, value in structure.items():
                    csv_rows.append([key, value.replace('\n', ' ') if value else ''])
                csv_rows.append(['', ''])
            
            # 주제 & 키워드
            keywords_themes = data.get('keywords_themes', {})
            if keywords_themes and 'error' not in keywords_themes:
                csv_rows.append(['=== 주제 & 키워드 ===', ''])
                for key, value in keywords_themes.items():
                    csv_rows.append([key, value.replace('\n', ' | ') if value else ''])
                csv_rows.append(['', ''])
            
            # 참고문헌
            references = data.get('references', {})
            if references and 'error' not in references:
                csv_rows.append(['=== 참고문헌 분석 ===', ''])
                for key, value in references.items():
                    csv_rows.append([key, value.replace('\n', ' | ') if value else ''])
            
            # CSV 문자열 생성
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerows(csv_rows)
            return output.getvalue().encode('utf-8-sig')  # BOM 추가로 한글 깨짐 방지
        
        with col2:
            csv_data = generate_csv_data()
            st.download_button(
                label="📥 CSV 다운로드",
                data=csv_data,
                file_name=f"{selected_paper}_분석결과.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        if meta['title'] or meta['author']:
            with st.expander("📋 문서 정보", expanded=False):
                cols = st.columns(4)
                if meta['title']:
                    cols[0].metric("제목", meta['title'][:50] + "..." if len(meta['title']) > 50 else meta['title'])
                if meta['author']:
                    cols[1].metric("저자", meta['author'][:30] + "..." if len(meta['author']) > 30 else meta['author'])
                if meta['pages']:
                    cols[2].metric("페이지", meta['pages'])
                if meta['creator']:
                    cols[3].metric("작성 도구", meta['creator'][:30] if meta['creator'] else 'N/A')
        
        # 탭 생성 (논문 2개 이상일 때 비교 탭 추가)
        if len(st.session_state.papers) > 1:
            tabs = st.tabs([
                "🤖 종합 분석",
                "📊 구조 분석",
                "🎯 주제 & 키워드",
                "📚 참고문헌",
                "🔄 논문 비교"
            ])
        else:
            tabs = st.tabs([
                "🤖 종합 분석",
                "📊 구조 분석",
                "🎯 주제 & 키워드",
                "📚 참고문헌"
            ])
        
        # 탭 1: 종합 분석
        with tabs[0]:
            st.markdown('<div class="section-header">🤖 종합 분석</div>', unsafe_allow_html=True)
            st.caption("🔹 논문의 핵심 내용을 체계적으로 분석합니다")
            
            analysis = data.get('main_analysis', {})
            
            if 'error' in analysis:
                st.error(analysis['error'])
            else:
                # 핵심요약 - 더 눈에 띄게 표시
                if '핵심요약' in analysis and analysis['핵심요약']:
                    st.markdown("### 📝 핵심 요약")
                    st.markdown(f"""<div style="background-color: #e8f4f8; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4;">
                    <h4 style="margin-top: 0;">요약</h4>
                    <p style="font-size: 16px; line-height: 1.6;">{analysis['핵심요약']}</p>
                    </div>""", unsafe_allow_html=True)
                    st.markdown("---")
                
                # 2컬럼 레이아웃
                col1, col2 = st.columns(2)
                
                with col1:
                    if '연구목적' in analysis and analysis['연구목적']:
                        st.markdown("### 🎯 연구 목적")
                        st.markdown(f"<div style='padding: 15px; background-color: #f0f8ff; border-radius: 8px;'>{analysis['연구목적']}</div>", unsafe_allow_html=True)
                        st.markdown("")
                    
                    if '연구방법' in analysis and analysis['연구방법']:
                        st.markdown("### 🔬 연구 방법")
                        st.markdown(f"<div style='padding: 15px; background-color: #f5f5f5; border-radius: 8px;'>{analysis['연구방법']}</div>", unsafe_allow_html=True)
                        st.markdown("")
                    
                    if '이론적기여' in analysis and analysis['이론적기여']:
                        st.markdown("### 💡 이론적 기여")
                        st.markdown(f"<div style='padding: 15px; background-color: #fff8dc; border-radius: 8px;'>{analysis['이론적기여']}</div>", unsafe_allow_html=True)
                
                with col2:
                    if '주요발견' in analysis and analysis['주요발견']:
                        st.markdown("### 🔍 주요 발견")
                        st.markdown(f"<div style='padding: 15px; background-color: #f0fff0; border-radius: 8px;'>{analysis['주요발견']}</div>", unsafe_allow_html=True)
                        st.markdown("")
                    
                    if '실무적시사점' in analysis and analysis['실무적시사점']:
                        st.markdown("### 📊 실무적 시사점")
                        st.markdown(f"<div style='padding: 15px; background-color: #fffacd; border-radius: 8px;'>{analysis['실무적시사점']}</div>", unsafe_allow_html=True)
                        st.markdown("")
                    
                    if '한계점' in analysis and analysis['한계점']:
                        st.markdown("### ⚠️ 연구 한계 및 향후 방향")
                        st.markdown(f"<div style='padding: 15px; background-color: #ffe4e1; border-radius: 8px;'>{analysis['한계점']}</div>", unsafe_allow_html=True)
        
        # 탭 2: 구조 분석
        with tabs[1]:
            st.markdown('<div class="section-header">📊 논문 구조 분석</div>', unsafe_allow_html=True)
            st.caption("🔹 IMRaD 구조에 따라 논문을 체계적으로 분해합니다")
            
            structure = data.get('structure', {})
            
            if 'error' in structure:
                st.error(structure['error'])
            else:
                # 각 섹션 표시
                sections = [
                    ("서론_배경", "📖 서론 및 배경", "#e8f4f8"),
                    ("이론적_프레임워크", "🎓 이론적 프레임워크", "#f0f8ff"),
                    ("연구방법", "🔬 연구방법", "#f5f5f5"),
                    ("자료분석", "📊 자료분석 방법", "#fff8dc"),
                    ("연구결과", "🔍 주요 연구결과", "#f0fff0"),
                    ("논의_함의", "💬 논의 및 함의", "#fffacd")
                ]
                
                for key, title, bg_color in sections:
                    if key in structure and structure[key]:
                        st.markdown(f"### {title}")
                        st.markdown(f"""<div style="padding: 15px; background-color: {bg_color}; border-radius: 8px; margin-bottom: 15px;">
                        {structure[key]}
                        </div>""", unsafe_allow_html=True)
        
        # 탭 3: 주제 & 키워드
        with tabs[2]:
            st.markdown('<div class="section-header">🎯 주제 & 키워드 분석</div>', unsafe_allow_html=True)
            st.caption("🔹 연구질문, 핵심개념, 키워드를 추출하고 관계를 시각화합니다")
            
            keywords_themes = data.get('keywords_themes', {})
            
            if 'error' in keywords_themes:
                st.error(keywords_themes['error'])
            else:
                # 연구질문
                if '연구질문' in keywords_themes and keywords_themes['연구질문']:
                    st.markdown("### ❓ 연구질문")
                    rqs = keywords_themes['연구질문'].strip().split('\n')
                    for rq in rqs:
                        rq = rq.strip()
                        if rq and (rq.startswith('•') or rq.startswith('-') or rq.startswith('*')):
                            rq = rq[1:].strip()
                        if rq:
                            st.markdown(f"""<div style="padding: 10px; background-color: #e8f4f8; border-left: 4px solid #1f77b4; margin-bottom: 8px; border-radius: 5px;">
                            <b>RQ:</b> {rq}
                            </div>""", unsafe_allow_html=True)
                    st.markdown("---")
                
                # 연구가설
                if '연구가설' in keywords_themes and keywords_themes['연구가설']:
                    st.markdown("### 💭 연구가설")
                    hyps = keywords_themes['연구가설'].strip().split('\n')
                    for hyp in hyps:
                        hyp = hyp.strip()
                        if hyp and (hyp.startswith('•') or hyp.startswith('-') or hyp.startswith('*')):
                            hyp = hyp[1:].strip()
                        if hyp:
                            st.markdown(f"""<div style="padding: 10px; background-color: #f0f8ff; border-left: 4px solid #4682b4; margin-bottom: 8px; border-radius: 5px;">
                            <b>H:</b> {hyp}
                            </div>""", unsafe_allow_html=True)
                    st.markdown("---")
                
                # 주요주제
                if '주요주제' in keywords_themes and keywords_themes['주요주제']:
                    st.markdown("### 🏷️ 주요 주제")
                    themes = [t.strip() for t in keywords_themes['주요주제'].strip().split('\n') if t.strip()]
                    # 불릿 마크 제거
                    themes = [t[1:].strip() if t.startswith(('•', '-', '*')) else t for t in themes]
                    
                    cols = st.columns(min(3, len(themes)))
                    for i, theme in enumerate(themes):
                        if theme:
                            cols[i % len(cols)].markdown(f"""<div style="padding: 15px; background-color: #fff8dc; border-radius: 8px; text-align: center; height: 100px; display: flex; align-items: center; justify-content: center;">
                            <b>{theme}</b>
                            </div>""", unsafe_allow_html=True)
                    st.markdown("---")
                
                # 핵심개념 & 중요키워드 2컬럼
                col1, col2 = st.columns(2)
                
                with col1:
                    if '핵심개념' in keywords_themes and keywords_themes['핵심개념']:
                        st.markdown("### 🧩 핵심 개념")
                        concepts = [c.strip() for c in keywords_themes['핵심개념'].replace(',', '\n').split('\n') if c.strip()]
                        concepts = [c[1:].strip() if c.startswith(('•', '-', '*')) else c for c in concepts]
                        for i, concept in enumerate(concepts[:10], 1):
                            if concept:
                                st.markdown(f"`{i}.` **{concept}**")
                
                with col2:
                    if '중요키워드' in keywords_themes and keywords_themes['중요키워드']:
                        st.markdown("### 🔑 중요 키워드")
                        keywords = [k.strip() for k in keywords_themes['중요키워드'].replace(',', '\n').split('\n') if k.strip()]
                        keywords = [k[1:].strip() if k.startswith(('•', '-', '*')) else k for k in keywords]
                        for i, keyword in enumerate(keywords[:10], 1):
                            if keyword:
                                st.markdown(f"`{i}.` **{keyword}**")
                
                # 학술용어
                if '학술용어' in keywords_themes and keywords_themes['학술용어']:
                    st.markdown("---")
                    st.markdown("### 🎓 학술 용어")
                    terms = [t.strip() for t in keywords_themes['학술용어'].replace(',', '\n').split('\n') if t.strip()]
                    terms = [t[1:].strip() if t.startswith(('•', '-', '*')) else t for t in terms]
                    st.markdown(" • ".join(terms[:15]))
                
                # 키워드 개념도 시각화
                st.markdown("---")
                st.markdown("### 🗺️ 키워드 개념도")
                st.markdown("""
                <div style="padding: 12px; background-color: #f0f8ff; border-left: 4px solid #2196F3; border-radius: 5px; margin-bottom: 15px;">
                📘 <b>개념도 설명</b><br>
                이 그래프는 논문의 핵심 주제와 관련 키워드 간의 관계를 시각화합니다.<br>
                • <span style="color: #FF6B6B;">⬤ 빨간색 노드</span>: 논문의 핵심 주제 (중심 개념)<br>
                • <span style="color: #4ECDC4;">⬤ 청록색 노드</span>: 관련 키워드 및 하위 개념<br>
                • <b>선(edge)</b>: 주제와 키워드 간의 연관성을 나타냅니다.<br>
                💡 이 시각화를 통해 논문의 이론적 구조와 개념 간 관계를 한눈에 파악할 수 있습니다.
                </div>
                """, unsafe_allow_html=True)
                
                # 모든 키워드 수집
                all_keywords = []
                if '핵심개념' in keywords_themes and keywords_themes['핵심개념']:
                    concepts = [c.strip() for c in keywords_themes['핵심개념'].replace(',', '\n').split('\n') if c.strip()]
                    all_keywords.extend([c[1:].strip() if c.startswith(('•', '-', '*')) else c for c in concepts])
                if '중요키워드' in keywords_themes and keywords_themes['중요키워드']:
                    keywords_list = [k.strip() for k in keywords_themes['중요키워드'].replace(',', '\n').split('\n') if k.strip()]
                    all_keywords.extend([k[1:].strip() if k.startswith(('•', '-', '*')) else k for k in keywords_list])
                
                if len(all_keywords) >= 3:
                    # 키워드 네트워크 생성
                    G = nx.Graph()
                    
                    # 중심 노드
                    if '주요주제' in keywords_themes and keywords_themes['주요주제']:
                        main_themes = [t.strip() for t in keywords_themes['주요주제'].strip().split('\n') if t.strip()]
                        main_theme = main_themes[0][1:].strip() if main_themes[0].startswith(('•', '-', '*')) else main_themes[0]
                        G.add_node(main_theme, node_type='main', size=30)
                        
                        # 키워드를 중심 주제와 연결
                        for i, kw in enumerate(all_keywords[:12]):
                            if kw and kw != main_theme:
                                G.add_node(kw, node_type='keyword', size=15)
                                G.add_edge(main_theme, kw)
                    else:
                        # 주제가 없으면 첫 키워드를 중심으로
                        if all_keywords:
                            G.add_node(all_keywords[0], node_type='main', size=30)
                            for kw in all_keywords[1:12]:
                                if kw:
                                    G.add_node(kw, node_type='keyword', size=15)
                                    G.add_edge(all_keywords[0], kw)
                    
                    if len(G.nodes()) > 1:
                        # 레이아웃 계산
                        pos = nx.spring_layout(G, k=2, iterations=50)
                        
                        # 엣지 트레이스
                        edge_trace = go.Scatter(
                            x=[], y=[],
                            line=dict(width=1, color='#888'),
                            hoverinfo='none',
                            mode='lines')
                        
                        for edge in G.edges():
                            x0, y0 = pos[edge[0]]
                            x1, y1 = pos[edge[1]]
                            edge_trace['x'] += tuple([x0, x1, None])
                            edge_trace['y'] += tuple([y0, y1, None])
                        
                        # 노드 트레이스
                        node_trace = go.Scatter(
                            x=[], y=[],
                            text=[],
                            mode='markers+text',
                            hoverinfo='text',
                            marker=dict(
                                showscale=False,
                                size=[],
                                color=[],
                                line_width=2))
                        
                        for node in G.nodes():
                            x, y = pos[node]
                            node_trace['x'] += tuple([x])
                            node_trace['y'] += tuple([y])
                            node_trace['text'] += tuple([node])
                            node_trace['marker']['size'] += tuple([G.nodes[node].get('size', 15)])
                            node_trace['marker']['color'] += tuple(['#FF6B6B' if G.nodes[node].get('node_type') == 'main' else '#4ECDC4'])
                        
                        # 그래프 생성
                        fig = go.Figure(data=[edge_trace, node_trace],
                                      layout=go.Layout(
                                          showlegend=False,
                                          hovermode='closest',
                                          margin=dict(b=0,l=0,r=0,t=0),
                                          xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                          yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                          height=500,
                                          plot_bgcolor='rgba(0,0,0,0)',
                                          paper_bgcolor='rgba(0,0,0,0)'
                                      ))
                        
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption("💡 중심 노드(빨강)는 핵심 주제, 주변 노드(청록)는 관련 키워드를 나타냅니다.")
                else:
                    st.info("키워드가 충분하지 않아 개념도를 생성할 수 없습니다.")
        
        # 탭 4: 참고문헌
        with tabs[3]:
            st.markdown('<div class="section-header">📚 참고문헌 분석</div>', unsafe_allow_html=True)
            st.caption("🔹 핵심 문헌을 파악하고 인용 관계를 시각화합니다")
            
            refs = data.get('references', {})
            
            if 'error' in refs:
                st.warning(refs.get('error', '참고문헌 분석을 수행할 수 없습니다.'))
            else:
                # 통계요약
                if '통계요약' in refs and refs['통계요약']:
                    st.markdown("### 📊 통계 요약")
                    st.markdown(f"""<div style="padding: 15px; background-color: #f0f8ff; border-radius: 8px; margin-bottom: 20px;">
                    {refs['통계요약'].replace(chr(10), '<br>')}
                    </div>""", unsafe_allow_html=True)
                
                # 핵심문헌 (가장 중요!)
                if '핵심문헌' in refs and refs['핵심문헌']:
                    st.markdown("### 📖 핵심 문헌 (필독)")
                    st.markdown("""<div style="background-color: #fffacd; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    💡 <b>연구에 가장 중요한 참고문헌들입니다. 각 문헌의 추천 사유를 확인하세요.</b>
                    </div>""", unsafe_allow_html=True)
                    
                    # 참고문헌을 파싱 (문헌 정보와 추천 사유 분리)
                    core_refs_text = refs['핵심문헌'].strip().split('\n')
                    
                    ref_counter = 0
                    current_ref = None
                    current_reasons = []
                    
                    for line in core_refs_text:
                        line = line.strip()
                        if not line:
                            continue
                            
                        # 새로운 문헌 시작 (• 또는 - 또는 * 로 시작)
                        if line.startswith(('• ', '- ', '* ')) and not line.startswith(('• →', '- →', '* →')):
                            # 이전 문헌 출력
                            if current_ref:
                                ref_counter += 1
                                st.markdown(f"""<div style="padding: 15px; background-color: #ffffff; border-left: 4px solid #4CAF50; margin-bottom: 15px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                                <b style="color: #4CAF50; font-size: 16px;">[{ref_counter}]</b> <span style="font-size: 15px;">{current_ref}</span>
                                """, unsafe_allow_html=True)
                                
                                if current_reasons:
                                    st.markdown('<div style="margin-top: 10px; padding-left: 10px; border-left: 2px solid #E0E0E0;">', unsafe_allow_html=True)
                                    for reason in current_reasons:
                                        reason = reason.strip()
                                        if '[사실]' in reason or '[추론]' in reason:
                                            # 사실과 추론에 색상 적용
                                            if '[사실]' in reason:
                                                reason_colored = reason.replace('[사실]', '<span style="color: #2196F3; font-weight: bold;">📌 사실:</span>')
                                            elif '[추론]' in reason:
                                                reason_colored = reason.replace('[추론]', '<span style="color: #FF9800; font-weight: bold;">💭 추론:</span>')
                                            st.markdown(f'<p style="margin: 5px 0; font-size: 14px;">{reason_colored}</p>', unsafe_allow_html=True)
                                    st.markdown('</div></div>', unsafe_allow_html=True)
                                else:
                                    st.markdown('</div>', unsafe_allow_html=True)
                            
                            # 새 문헌 시작
                            current_ref = line[2:].strip()  # • 또는 - 제거
                            current_reasons = []
                        
                        # 추천 사유 (→ 로 시작)
                        elif line.startswith('→') or line.startswith('• →') or line.startswith('- →') or line.startswith('* →'):
                            current_reasons.append(line.replace('• →', '→').replace('- →', '→').replace('* →', '→').strip())
                    
                    # 마지막 문헌 출력
                    if current_ref:
                        ref_counter += 1
                        st.markdown(f"""<div style="padding: 15px; background-color: #ffffff; border-left: 4px solid #4CAF50; margin-bottom: 15px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                        <b style="color: #4CAF50; font-size: 16px;">[{ref_counter}]</b> <span style="font-size: 15px;">{current_ref}</span>
                        """, unsafe_allow_html=True)
                        
                        if current_reasons:
                            st.markdown('<div style="margin-top: 10px; padding-left: 10px; border-left: 2px solid #E0E0E0;">', unsafe_allow_html=True)
                            for reason in current_reasons:
                                reason = reason.strip()
                                if '[사실]' in reason or '[추론]' in reason:
                                    if '[사실]' in reason:
                                        reason_colored = reason.replace('[사실]', '<span style="color: #2196F3; font-weight: bold;">📌 사실:</span>')
                                    elif '[추론]' in reason:
                                        reason_colored = reason.replace('[추론]', '<span style="color: #FF9800; font-weight: bold;">💭 추론:</span>')
                                    st.markdown(f'<p style="margin: 5px 0; font-size: 14px;">{reason_colored}</p>', unsafe_allow_html=True)
                            st.markdown('</div></div>', unsafe_allow_html=True)
                        else:
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown("---")
                
                # 2컬럼 레이아웃
                col1, col2 = st.columns(2)
                
                with col1:
                    # 주요저널
                    if '주요저널' in refs and refs['주요저널']:
                        st.markdown("### 📰 주요 저널")
                        journals = [j.strip() for j in refs['주요저널'].strip().split('\n') if j.strip()]
                        journals = [j[1:].strip() if j.startswith(('•', '-', '*')) else j for j in journals]
                        for journal in journals[:5]:
                            if journal:
                                st.markdown(f"• {journal}")
                        st.markdown("")
                    
                    # 출판물유형
                    if '출판물유형' in refs and refs['출판물유형']:
                        st.markdown("### 📑 출판물 유형")
                        types = [t.strip() for t in refs['출판물유형'].strip().split('\n') if t.strip()]
                        types = [t[1:].strip() if t.startswith(('•', '-', '*')) else t for t in types]
                        for pub_type in types:
                            if pub_type:
                                st.markdown(f"• {pub_type}")
                
                with col2:
                    # 영향력있는연구자
                    if '영향력있는연구자' in refs and refs['영향력있는연구자']:
                        st.markdown("### 👨‍🔬 영향력 있는 연구자")
                        researchers = [r.strip() for r in refs['영향력있는연구자'].strip().split('\n') if r.strip()]
                        researchers = [r[1:].strip() if r.startswith(('•', '-', '*')) else r for r in researchers]
                        for researcher in researchers[:5]:
                            if researcher:
                                st.markdown(f"• {researcher}")
                
                # 시사점
                if '시사점' in refs and refs['시사점']:
                    st.markdown("---")
                    st.markdown("### 💡 문헌 분석 시사점")
                    st.markdown(f"""<div style="padding: 15px; background-color: #e8f5e9; border-radius: 8px; border-left: 5px solid #4CAF50;">
                    {refs['시사점']}
                    </div>""", unsafe_allow_html=True)
                
                # 인용 네트워크 시각화
                st.markdown("---")
                st.markdown("### 🔗 인용 네트워크")
                st.markdown("""
                <div style="padding: 12px; background-color: #fff3e0; border-left: 4px solid #FF9800; border-radius: 5px; margin-bottom: 15px;">
                📘 <b>인용 네트워크 설명</b> | 🐍 <i>Python (NetworkX) 기반 시각화</i><br>
                이 그래프는 논문의 참고문헌에 나타난 주요 연구자와 문헌 간의 관계를 시각화합니다.<br>
                • <span style="color: #FF6B6B;">⬤ 빨간색 노드</span>: 영향력 있는 연구자 (인용 횟수가 많은 저자)<br>
                • <span style="color: #95E1D3;">⬤ 청록색 노드</span>: 핵심 참고문헌 (주요 논문)<br>
                • <b>선(edge)</b>: 저자-논문 간의 저작 관계를 나타냅니다.<br>
                💡 이 시각화를 통해 연구 분야의 주요 학자와 그들의 핵심 저작물을 파악할 수 있습니다.<br>
                ⚠️ <i>주의: 네트워크 연결은 저자명 유사도 기반으로 추정되므로 실제와 다를 수 있습니다.</i>
                </div>
                """, unsafe_allow_html=True)
                
                # 핵심문헌과 연구자 정보로 네트워크 생성
                if '핵심문헌' in refs and refs['핵심문헌'] and '영향력있는연구자' in refs and refs['영향력있는연구자']:
                    G = nx.Graph()
                    
                    # 핵심문헌에서 저자 추출 (간단하게 파싱)
                    core_refs = [r.strip() for r in refs['핵심문헌'].strip().split('\n') if r.strip()]
                    core_refs = [r[1:].strip() if r.startswith(('•', '-', '*')) else r for r in core_refs]
                    
                    researchers = [r.strip() for r in refs['영향력있는연구자'].strip().split('\n') if r.strip()]
                    researchers = [r[1:].strip() if r.startswith(('•', '-', '*')) else r for r in researchers]
                    
                    # 연구자 노드 추가
                    for researcher in researchers[:5]:
                        if researcher and '(' in researcher:
                            author_name = researcher.split('(')[0].strip()
                            if author_name:
                                G.add_node(author_name, node_type='author', size=25)
                    
                    # 문헌 노드 추가 및 연결
                    for i, ref in enumerate(core_refs[:6]):
                        if ref:
                            # 저자명 추출 시도 (첫 단어 또는 괄호 전까지)
                            parts = ref.split('(')
                            if len(parts) > 1:
                                author_from_ref = parts[0].strip().split()[0] if parts[0].strip() else f"문헌{i+1}"
                            else:
                                author_from_ref = f"문헌{i+1}"
                            
                            # 노드에 전체 참조를 저장 (display용과 hover용 분리)
                            G.add_node(ref, node_type='paper', size=15, full_ref=ref)
                            
                            # 저자와 문헌 연결 (이름이 유사하면)
                            for author_node in [n for n in G.nodes() if G.nodes[n].get('node_type') == 'author']:
                                if any(word in author_from_ref.lower() for word in author_node.lower().split()[:2]):
                                    G.add_edge(author_node, ref)
                    
                    # 문헌 간 연결 (같은 저자가 쓴 것으로 추정)
                    papers = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'paper']
                    for i, paper1 in enumerate(papers):
                        for paper2 in papers[i+1:i+3]:  # 인접한 2개만 연결
                            if nx.has_path(G, paper1, paper2) and nx.shortest_path_length(G, paper1, paper2) == 2:
                                continue  # 이미 공통 저자로 연결됨
                    
                    if len(G.nodes()) > 2:
                        # 레이아웃 계산
                        pos = nx.spring_layout(G, k=3, iterations=50)
                        
                        # 엣지 트레이스
                        edge_trace = go.Scatter(
                            x=[], y=[],
                            line=dict(width=0.5, color='#888'),
                            hoverinfo='none',
                            mode='lines')
                        
                        for edge in G.edges():
                            x0, y0 = pos[edge[0]]
                            x1, y1 = pos[edge[1]]
                            edge_trace['x'] += tuple([x0, x1, None])
                            edge_trace['y'] += tuple([y0, y1, None])
                        
                        # 노드 트레이스
                        node_trace = go.Scatter(
                            x=[], y=[],
                            text=[],
                            hovertext=[],
                            mode='markers+text',
                            hoverinfo='text',
                            textposition='top center',
                            marker=dict(
                                showscale=False,
                                size=[],
                                color=[],
                                line_width=2))
                        
                        for node in G.nodes():
                            x, y = pos[node]
                            node_trace['x'] += tuple([x])
                            node_trace['y'] += tuple([y])
                            
                            node_type = G.nodes[node].get('node_type', 'paper')
                            
                            # 노드 라벨 (display용 - 짧게)
                            if node_type == 'paper':
                                label = node[:50] + "..." if len(node) > 50 else node
                            else:
                                label = node
                            node_trace['text'] += tuple([label])
                            
                            # Hover 정보 (전체 이름)
                            hover_text = node  # 전체 이름 표시
                            node_trace['hovertext'] += tuple([hover_text])
                            
                            node_trace['marker']['size'] += tuple([G.nodes[node].get('size', 15)])
                            node_trace['marker']['color'] += tuple(['#FF6B6B' if node_type == 'author' else '#95E1D3'])
                        
                        # 그래프 생성
                        fig = go.Figure(data=[edge_trace, node_trace],
                                      layout=go.Layout(
                                          showlegend=False,
                                          hovermode='closest',
                                          margin=dict(b=0,l=0,r=0,t=40),
                                          xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                          yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                          height=600,
                                          plot_bgcolor='rgba(0,0,0,0)',
                                          paper_bgcolor='rgba(0,0,0,0)'
                                      ))
                        
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption("💡 빨간 노드는 영향력 있는 연구자, 청록 노드는 핵심 문헌을 나타냅니다. 선은 저자-논문 관계를 표시합니다.")
                    else:
                        st.info("네트워크를 생성하기에 충분한 정보가 없습니다.")
                else:
                    st.info("핵심문헌 또는 연구자 정보가 없어 네트워크를 생성할 수 없습니다.")
        
        # 탭 5: 논문 비교 (2개 이상일 때만)
        if len(st.session_state.papers) > 1:
            with tabs[4]:
                st.markdown('<div class="section-header">🔄 논문 비교 분석</div>', unsafe_allow_html=True)
                st.caption("🔹 두 논문을 비교하여 연구 공백, 방법론 차이, 이론적 차이를 분석합니다")
                
                # 비교할 논문 선택
                st.markdown("### 📝 비교할 논문 선택")
                paper_names = list(st.session_state.papers.keys())
                
                col1, col2 = st.columns(2)
                with col1:
                    paper1 = st.selectbox("논문 1", paper_names, key="compare_paper1")
                with col2:
                    paper2_options = [p for p in paper_names if p != paper1]
                    paper2 = st.selectbox("논문 2", paper2_options, key="compare_paper2") if paper2_options else None
                
                if paper2:
                    st.markdown("---")
                    
                    # 세션에 비교 결과 저장 키 생성
                    comparison_key = f"{paper1}_vs_{paper2}"
                    
                    # 비교 분석 실행 버튼
                    if st.button("🚀 심층 비교 분석 시작", type="primary", use_container_width=True):
                        with st.spinner("📊 논문을 비교 분석 중입니다... (약 20-30초 소요)"):
                            # GPT 비교 분석
                            compare_data = {
                                paper1: st.session_state.papers[paper1],
                                paper2: st.session_state.papers[paper2]
                            }
                            comparison = gpt_compare_papers(compare_data)
                            
                            # 세션에 결과 저장
                            if 'comparisons' not in st.session_state:
                                st.session_state.comparisons = {}
                            st.session_state.comparisons[comparison_key] = comparison
                            st.rerun()
                    
                    # 저장된 비교 결과 표시
                    if 'comparisons' in st.session_state and comparison_key in st.session_state.comparisons:
                        comparison = st.session_state.comparisons[comparison_key]
                        
                        if 'error' not in comparison:
                            st.success("✅ 비교 분석 완료!")
                            
                            # 연구 공백
                            if '연구공백' in comparison:
                                st.markdown("### 🎯 연구 공백 (Research Gap)")
                                st.markdown("""
                                <div style="background-color: #e8f5e9; padding: 15px; border-radius: 8px; border-left: 5px solid #4CAF50; margin-bottom: 15px;">
                                💡 <b>활용 방법:</b> 이 정보를 활용하여 새로운 연구 주제를 선정하거나 연구 제안서의 차별성을 강조할 수 있습니다.
                                </div>
                                """, unsafe_allow_html=True)
                                st.markdown(comparison['연구공백'])
                                st.markdown("---")
                            
                            # 방법론 비교
                            if '방법론비교' in comparison:
                                st.markdown("### 🔬 방법론 비교")
                                st.markdown("""
                                <div style="background-color: #fff3e0; padding: 15px; border-radius: 8px; border-left: 5px solid #FF9800; margin-bottom: 15px;">
                                💡 <b>활용 방법:</b> 각 방법론의 장단점을 이해하고 자신의 연구 상황에 적합한 방법을 선택하세요.
                                </div>
                                """, unsafe_allow_html=True)
                                st.markdown(comparison['방법론비교'])
                                st.markdown("---")
                            
                            # 이론적 차이
                            if '이론적차이' in comparison:
                                st.markdown("### 📚 이론적 프레임워크 비교")
                                st.markdown("""
                                <div style="background-color: #f3e5f5; padding: 15px; border-radius: 8px; border-left: 5px solid #9C27B0; margin-bottom: 15px;">
                                💡 <b>활용 방법:</b> 문헌고찰에서 이론 비교 섹션을 작성하거나 자신의 연구에 적합한 이론을 선택하세요.
                                </div>
                                """, unsafe_allow_html=True)
                                st.markdown(comparison['이론적차이'])
                                st.markdown("---")
                            
                            # 주요 차별점
                            if '주요차별점' in comparison:
                                st.markdown("### 🔍 주요 차별점")
                                st.info(comparison['주요차별점'])
                                st.markdown("---")
                            
                            # 연구 제안
                            if '연구제안' in comparison:
                                st.markdown("### 💡 새로운 연구 제안")
                                st.markdown("""
                                <div style="background-color: #e3f2fd; padding: 15px; border-radius: 8px; border-left: 5px solid #2196F3; margin-bottom: 15px;">
                                💡 <b>활용 방법:</b> 이 제안을 바탕으로 연구 계획서를 작성하거나 지도교수와 논의할 수 있습니다.
                                </div>
                                """, unsafe_allow_html=True)
                                st.success(comparison['연구제안'])
                        else:
                            st.error(comparison['error'])
                    else:
                        st.info("💡 '심층 비교 분석 시작' 버튼을 클릭하여 분석을 시작하세요.")
                else:
                    st.warning("비교할 두 번째 논문을 선택해주세요.")

if __name__ == "__main__":
    main()
