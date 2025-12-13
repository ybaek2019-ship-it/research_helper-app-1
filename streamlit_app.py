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
import pandas as pd
import csv
import io

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
                {"role": "system", "content": "당신은 학술 논문 분석 전문가입니다. 질적 연구방법론에 특히 정통하며, 한국어로 명확하고 상세한 분석을 제공합니다."},
                {"role": "user", "content": f"""다음 학술 논문을 종합적으로 분석하여 한국어로 답변해주세요:

{truncated_text}

다음 섹션별로 명확하게 구분하여 작성해주세요:

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
                {"role": "system", "content": "당신은 학술 논문의 구조를 분석하는 전문가입니다. IMRaD 구조(서론, 방법, 결과, 논의)를 잘 이해하고 있습니다."},
                {"role": "user", "content": f"""다음 논문의 구조를 분석하여 각 섹션을 요약해주세요:

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
                {"role": "system", "content": "당신은 학술 논문의 주제와 키워드를 추출하는 전문가입니다."},
                {"role": "user", "content": f"""다음 논문에서 연구질문, 주요 주제, 키워드를 추출해주세요:

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
                {"role": "system", "content": "당신은 학술 논문의 참고문헌을 분석하는 전문가입니다. 서지정보를 정확히 추출하고 대학원생에게 유용한 인사이트를 제공합니다."},
                {"role": "user", "content": f"""다음 참고문헌 목록을 분석하여 대학원생이 문헌 조사에 활용할 수 있도록 상세히 정리해주세요:

{ref_section}

다음 형식으로 작성해주세요:

[통계요약]
• 총 참고문헌: XX개
• 연도 범위: XXXX-XXXX년
• 최근 5년 이내: XX개 (XX%)
• 평균 저자수: X.X명

[핵심문헌]
각 문헌을 다음 형식으로 나열 (최대 8개):
• 저자(연도). 제목. 저널/출판사. (피인용 횟수가 많거나 핵심적인 문헌 위주)

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

# 고급분석 및 비교분석 기능 제거됨 (안정성 향상을 위해)
# 핵심 분석 기능에만 집중: 종합분석, 구조분석, 주제&키워드 분석, 참고문헌 분석

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
    
    st.markdown('<div class="main-header">📚 AI 학술 논문 분석 도구</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI 기반 대학원생을 위한 지능형 학술논문 분석 시스템</div>', unsafe_allow_html=True)
    
    if 'papers' not in st.session_state:
        st.session_state.papers = {}
    
    # 사이드바
    with st.sidebar:
        st.header("📤 PDF 업로드")
        
        with st.expander("ℹ️ 사용 가이드", expanded=False):
            st.markdown("""
            **🤖 AI 기반 분석 기능:**
            - AI 종합 분석 (요약, 주제, 키워드)
            - 구조 분석 (서론, 방법, 결과, 논의)
            - 참고문헌 심층 분석
            - 고급 텍스트 분석 (가독성, 담화 구조)
            - 다중 논문 비교 분석
            
            **📁 파일 크기:**
            - 권장: 10MB 이하
            - 최대: 30MB
            - 20MB 이상: 압축 권장
            
            **💡 모든 분석이 AI로 수행됩니다.**
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
        
        analyze_button = st.button("🔍 AI 분석 시작", type="primary", use_container_width=True)
        
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
                                
                                status_text.text("🤖 AI 종합 분석 중...")
                                progress_bar.progress(20)
                                main_analysis = gpt_analyze_all(text)
                                
                                status_text.text("🤖 AI 구조 분석 중...")
                                progress_bar.progress(40)
                                structure = gpt_analyze_structure(text)
                                
                                status_text.text("🤖 AI 주제&키워드 분석 중...")
                                progress_bar.progress(60)
                                keywords_themes = gpt_analyze_keywords_themes(text)
                                
                                status_text.text("🤖 AI 참고문헌 분석 중...")
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
                                status_text.text("✅ AI 분석 완료!")
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
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 🤖 AI 종합 분석")
            st.write("AI가 논문을 읽고 핵심 내용, 주제, 키워드를 추출합니다.")
        with col2:
            st.markdown("### 📊 구조 분석")
            st.write("서론, 방법, 결과, 논의 등 논문 구조를 AI가 분석합니다.")
        with col3:
            st.markdown("### 📚 참고문헌 분석")
            st.write("AI가 참고문헌을 분석하여 연구 동향을 파악합니다.")
        
        st.markdown("---")
        st.markdown('<p style="text-align: center; color: #888; font-size: 0.85rem; margin-top: 2rem;">본 분석 도구는 GPT-4를 활용하여 학술 논문을 분석합니다.</p>', unsafe_allow_html=True)
    
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
        
        tabs = st.tabs([
            "🤖 종합 분석",
            "📊 구조 분석",
            "🎯 주제 & 키워드",
            "📚 참고문헌"
        ])
        
        # 탭 1: 종합 분석
        with tabs[0]:
            st.markdown('<div class="section-header">🤖 AI 종합 분석</div>', unsafe_allow_html=True)
            
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
        
        # 탭 4: 참고문헌
        with tabs[3]:
            st.markdown('<div class="section-header">📚 참고문헌 분석</div>', unsafe_allow_html=True)
            
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
                    💡 <b>연구에 가장 중요한 참고문헌들입니다. 문헌 조사 시 우선적으로 읽어보세요.</b>
                    </div>""", unsafe_allow_html=True)
                    
                    core_refs = [r.strip() for r in refs['핵심문헌'].strip().split('\n') if r.strip()]
                    core_refs = [r[1:].strip() if r.startswith(('•', '-', '*')) else r for r in core_refs]
                    
                    for i, ref in enumerate(core_refs, 1):
                        if ref:
                            st.markdown(f"""<div style="padding: 12px; background-color: #ffffff; border-left: 4px solid #4CAF50; margin-bottom: 10px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                            <b style="color: #4CAF50;">[{i}]</b> {ref}
                            </div>""", unsafe_allow_html=True)
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

if __name__ == "__main__":
    main()
