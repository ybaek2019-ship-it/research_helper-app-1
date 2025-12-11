import streamlit as st
from io import BytesIO
from collections import Counter
from pypdf import PdfReader
import re
import unicodedata
from openai import OpenAI
import json
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# 상수 정의
MAX_FILE_SIZE_MB = 20  # Streamlit 기본값보다 안전하게 설정
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# API 키 관리
CONFIG_DIR = Path(__file__).parent / "config"
CONFIG_FILE = CONFIG_DIR / "api_keys.json"

def load_api_key():
    """API 키를 로드합니다."""
    # 1. 환경 변수에서 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key
    
    # 2. Streamlit secrets에서 확인
    try:
        if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
            return st.secrets['OPENAI_API_KEY']
    except:
        pass
    
    # 3. 설정 파일에서 확인
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                return config.get('openai_api_key')
    except:
        pass
    
    return None

def save_api_key(api_key):
    """API 키를 설정 파일에 저장합니다."""
    try:
        CONFIG_DIR.mkdir(exist_ok=True)
        config = {'openai_api_key': api_key}
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        st.error(f"API 키 저장 실패: {str(e)}")
        return False

# OpenAI 클라이언트 초기화
@st.cache_resource
def get_openai_client():
    """OpenAI 클라이언트를 초기화합니다."""
    api_key = load_api_key()
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

# ==================== GPT 기반 분석 함수 ====================
def gpt_summarize(text, max_words=3000):
    """GPT를 사용하여 논문을 요약합니다."""
    try:
        client = get_openai_client()
        if not client:
            return {"error": "OpenAI API 키가 설정되지 않았습니다. 설정에서 API 키를 입력해주세요."}
        
        # 텍스트 길이 제한 (토큰 제한 고려)
        words = text.split()
        truncated_text = ' '.join(words[:max_words])
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 학술 논문 분석 전문가입니다. 질적 연구방법론에 특히 정통합니다."},
                {"role": "user", "content": f"""다음 학술 논문을 분석하여 한국어로 답변해주세요:

{truncated_text}

다음 형식의 JSON으로 응답하세요:
{{
  "핵심요약": "3-5문장의 핵심 내용 요약",
  "연구목적": "연구의 목적과 배경",
  "연구방법": "사용된 연구방법론 (질적/양적/혼합 등)",
  "주요발견": "핵심 연구 결과",
  "이론적기여": "이론적/실천적 함의",
  "한계점": "연구의 한계점"
}}"""}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        
        result = response.choices[0].message.content
        # JSON 파싱 시도
        try:
            return json.loads(result)
        except:
            # JSON이 아니면 텍스트 그대로 반환
            return {"핵심요약": result}
    except Exception as e:
        return {"error": f"GPT 분석 실패: {str(e)}"}

def gpt_extract_themes(text, max_words=2000):
    """GPT를 사용하여 주제를 추출합니다."""
    try:
        client = get_openai_client()
        if not client:
            return {"error": "OpenAI API 키가 설정되지 않았습니다."}
        
        words = text.split()
        truncated_text = ' '.join(words[:max_words])
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 질적연구 코딩 전문가입니다."},
                {"role": "user", "content": f"""다음 텍스트에서 주요 주제(theme)를 추출하세요:

{truncated_text}

JSON 형식으로 응답:
{{
  "주요주제": ["주제1", "주제2", "주제3", "주제4", "주제5"],
  "핵심개념": ["개념1", "개념2", "개념3", "개념4", "개념5"]
}}"""}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        result = response.choices[0].message.content
        try:
            return json.loads(result)
        except:
            return {"주요주제": [], "핵심개념": []}
    except Exception as e:
        return {"error": f"주제 추출 실패: {str(e)}"}

def gpt_compare_papers(paper_texts, max_words_per_paper=1500):
    """GPT를 사용하여 여러 논문을 비교합니다."""
    try:
        client = get_openai_client()
        if not client:
            return {"error": "OpenAI API 키가 설정되지 않았습니다."}
        
        # 각 논문에서 일부만 추출
        truncated_papers = {}
        for name, text in paper_texts.items():
            words = text.split()
            truncated_papers[name] = ' '.join(words[:max_words_per_paper])
        
        papers_text = "\n\n".join([f"[논문: {name}]\n{text}" for name, text in truncated_papers.items()])
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 학술 논문 비교분석 전문가입니다."},
                {"role": "user", "content": f"""다음 논문들을 비교 분석하세요:

{papers_text}

JSON 형식으로 응답:
{{
  "공통주제": ["주제1", "주제2", "주제3"],
  "차별점": "각 논문의 주요 차별점",
  "방법론비교": "연구방법론의 유사점과 차이점",
  "종합평가": "전체적인 비교 평가"
}}"""}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        result = response.choices[0].message.content
        try:
            return json.loads(result)
        except:
            return {"종합평가": result}
    except Exception as e:
        return {"error": f"비교 분석 실패: {str(e)}"}

def gpt_research_questions(text, max_words=2000):
    """GPT를 사용하여 연구질문을 추출합니다."""
    try:
        client = get_openai_client()
        if not client:
            return {"error": "OpenAI API 키가 설정되지 않았습니다."}
        
        words = text.split()
        truncated_text = ' '.join(words[:max_words])
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 학술 논문의 연구질문 추출 전문가입니다."},
                {"role": "user", "content": f"""다음 논문에서 연구질문(Research Questions)을 추출하세요:

{truncated_text}

JSON 형식으로 응답:
{{
  "연구질문": ["RQ1", "RQ2", "RQ3"],
  "연구가설": ["H1", "H2"] 
}}

연구질문이 명시적으로 없다면 논문의 목적을 바탕으로 추론하세요."""}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        result = response.choices[0].message.content
        try:
            return json.loads(result)
        except:
            return {"연구질문": [], "연구가설": []}
    except Exception as e:
        return {"error": f"연구질문 추출 실패: {str(e)}"}
# 불용어 리스트 (확장)
STOP_WORDS = set([
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who',
    'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
    'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 'also', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off',
    'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
    'however', 'therefore', 'thus', 'furthermore', 'moreover', 'nevertheless',
    'although', 'though', 'whereas', 'while', 'since', 'because', 'unless',
    'whether', 'either', 'neither', 'rather', 'between', 'among', 'within'
])

# ==================== 고급 텍스트 분석 함수 ====================

def analyze_readability(text):
    """텍스트 가독성을 다양한 지표로 분석합니다."""
    try:
        # Flesch Reading Ease (0-100, 높을수록 읽기 쉬움)
        flesch_reading = textstat.flesch_reading_ease(text)
        
        # Flesch-Kincaid Grade Level (학년 수준)
        fk_grade = textstat.flesch_kincaid_grade(text)
        
        # SMOG Index (이해에 필요한 교육 연수)
        smog = textstat.smog_index(text)
        
        # Coleman-Liau Index
        coleman_liau = textstat.coleman_liau_index(text)
        
        # Automated Readability Index
        ari = textstat.automated_readability_index(text)
        
        # Dale-Chall Readability Score
        dale_chall = textstat.dale_chall_readability_score(text)
        
        # 평균값 계산
        avg_grade = (fk_grade + smog + coleman_liau + ari) / 4
        
        # 해석
        if flesch_reading >= 60:
            difficulty = "쉬움"
        elif flesch_reading >= 30:
            difficulty = "보통"
        else:
            difficulty = "어려움"
        
        return {
            'flesch_reading_ease': round(flesch_reading, 2),
            'flesch_kincaid_grade': round(fk_grade, 2),
            'smog_index': round(smog, 2),
            'coleman_liau': round(coleman_liau, 2),
            'ari': round(ari, 2),
            'dale_chall': round(dale_chall, 2),
            'average_grade_level': round(avg_grade, 2),
            'difficulty': difficulty
        }
    except Exception as e:
        return None

def analyze_sentence_complexity(text):
    """문장 복잡도를 분석합니다."""
    try:
        sentences = sent_tokenize(text)
        if not sentences:
            return None
        
        # 문장 길이 분석
        sentence_lengths = [len(word_tokenize(s)) for s in sentences]
        
        # 단어 길이 분석
        words = word_tokenize(text.lower())
        word_lengths = [len(w) for w in words if w.isalpha()]
        
        # 어휘 다양성 (Type-Token Ratio)
        unique_words = len(set(words))
        total_words = len(words)
        ttr = (unique_words / total_words * 100) if total_words > 0 else 0
        
        # 긴 단어 비율 (7자 이상)
        long_words = [w for w in words if len(w) >= 7 and w.isalpha()]
        long_word_ratio = (len(long_words) / len(words) * 100) if len(words) > 0 else 0
        
        return {
            'avg_sentence_length': round(np.mean(sentence_lengths), 2),
            'max_sentence_length': max(sentence_lengths),
            'min_sentence_length': min(sentence_lengths),
            'sentence_length_std': round(np.std(sentence_lengths), 2),
            'avg_word_length': round(np.mean(word_lengths), 2),
            'vocabulary_diversity': round(ttr, 2),
            'long_word_ratio': round(long_word_ratio, 2),
            'total_sentences': len(sentences),
            'total_words': len(words),
            'unique_words': unique_words
        }
    except Exception as e:
        return None

def extract_collocations(text, n=20):
    """통계적으로 유의미한 단어 조합(collocation)을 추출합니다."""
    try:
        # 텍스트 토큰화
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalpha() and len(w) > 3 and w not in STOP_WORDS]
        
        if len(words) < 20:
            return []
        
        # Bigram Collocation Finder
        bigram_measures = BigramAssocMeasures()
        finder = BigramCollocationFinder.from_words(words)
        
        # 최소 빈도 필터 (3번 이상 출현)
        finder.apply_freq_filter(3)
        
        # PMI (Pointwise Mutual Information) 기반 상위 collocation
        collocations = finder.nbest(bigram_measures.pmi, n)
        
        # 빈도수와 함께 반환
        collocation_freq = []
        for col in collocations:
            freq = finder.ngram_fd[col]
            collocation_freq.append((' '.join(col), freq))
        
        return collocation_freq
    except Exception as e:
        return []

def build_cooccurrence_network(text, top_n=30):
    """단어 공동 출현 네트워크를 구축합니다."""
    try:
        sentences = sent_tokenize(text)
        
        # 단어 공동 출현 행렬 생성
        cooccurrence = defaultdict(lambda: defaultdict(int))
        
        for sentence in sentences[:200]:  # 처음 200문장만 사용 (메모리 효율)
            words = word_tokenize(sentence.lower())
            words = [w for w in words if w.isalpha() and len(w) > 3 and w not in STOP_WORDS]
            
            # 같은 문장에 나타나는 단어 쌍
            for i, word1 in enumerate(words):
                for word2 in words[i+1:]:
                    cooccurrence[word1][word2] += 1
                    cooccurrence[word2][word1] += 1
        
        # 네트워크 그래프 생성
        G = nx.Graph()
        
        # 상위 빈도 단어 선택
        all_words = Counter()
        for word, cowords in cooccurrence.items():
            all_words[word] += sum(cowords.values())
        
        top_words = [word for word, _ in all_words.most_common(top_n)]
        
        # 엣지 추가
        for word1 in top_words:
            for word2 in top_words:
                if word1 != word2 and word2 in cooccurrence[word1]:
                    weight = cooccurrence[word1][word2]
                    if weight >= 2:  # 최소 2번 이상 공동 출현
                        G.add_edge(word1, word2, weight=weight)
        
        return G
    except Exception as e:
        return None

def extract_topics_lda(text, n_topics=5, n_words=10):
    """LDA 토픽 모델링을 수행합니다."""
    try:
        # 문장 분할
        sentences = sent_tokenize(text)
        
        if len(sentences) < 10:
            return None
        
        # TF-IDF 벡터화
        vectorizer = TfidfVectorizer(
            max_features=200,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # LDA 모델 학습
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=50,
            learning_method='batch'
        )
        
        lda.fit(tfidf_matrix)
        
        # 토픽별 상위 단어 추출
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(lda.components_):
            top_indices = topic.argsort()[-n_words:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            top_scores = [topic[i] for i in top_indices]
            
            topics.append({
                'topic_id': topic_idx + 1,
                'words': top_words,
                'scores': [round(float(s), 4) for s in top_scores]
            })
        
        return topics
    except Exception as e:
        return None

def calculate_semantic_similarity(text1, text2):
    """두 텍스트 간의 의미론적 유사도를 계산합니다."""
    try:
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        
        # TF-IDF 벡터화
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        
        # 코사인 유사도 계산
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return round(similarity * 100, 2)
    except Exception as e:
        return None

def analyze_discourse_markers(text):
    """담화 표지(discourse markers)를 분석합니다."""
    discourse_categories = {
        '인과관계': ['because', 'therefore', 'thus', 'hence', 'consequently', 'as a result', 'due to', 'since'],
        '대조': ['however', 'but', 'although', 'despite', 'nevertheless', 'on the other hand', 'whereas', 'while', 'yet'],
        '추가': ['furthermore', 'moreover', 'additionally', 'also', 'in addition', 'besides', 'likewise'],
        '예시': ['for example', 'for instance', 'such as', 'including', 'namely', 'specifically'],
        '결론': ['in conclusion', 'to conclude', 'in summary', 'to sum up', 'overall', 'finally'],
        '강조': ['indeed', 'in fact', 'actually', 'certainly', 'clearly', 'obviously']
    }
    
    text_lower = text.lower()
    results = {}
    
    for category, markers in discourse_categories.items():
        count = sum(text_lower.count(marker) for marker in markers)
        results[category] = count
    
    return results

def extract_citation_patterns(text):
    """인용 패턴을 분석합니다."""
    patterns = {
        'author_year': r'\([A-Z][a-z]+(?:\s+et al\.)?,?\s+\d{4}\)',
        'author_year_page': r'\([A-Z][a-z]+(?:\s+et al\.)?,?\s+\d{4},?\s+p+\.\s*\d+\)',
        'numbered': r'\[\d+\]',
        'multiple_authors': r'et al\.',
    }
    
    results = {}
    for pattern_name, pattern in patterns.items():
        matches = re.findall(pattern, text)
        results[pattern_name] = len(matches)
    
    return results

# ==================== 텍스트 전처리 ====================
def clean_text(text):
    """텍스트를 정제하고 정규화합니다."""
    # 유니코드 정규화
    text = unicodedata.normalize('NFKD', text)
    # 연속된 공백 제거
    text = re.sub(r'\s+', ' ', text)
    # 하이픈으로 나뉜 단어 복원
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    return text.strip()

# ==================== PDF 로드 ====================
def load_pdf_from_upload(uploaded_file):
    """업로드된 PDF 파일을 로드합니다."""
    try:
        # 파일 크기 확인
        file_size = uploaded_file.size
        file_size_mb = file_size / 1024 / 1024
        
        if file_size == 0:
            return None, "❌ 업로드된 파일이 비어있습니다. 올바른 PDF 파일을 선택해주세요."
        
        if file_size > MAX_FILE_SIZE_BYTES:
            return None, f"❌ 파일 크기가 {MAX_FILE_SIZE_MB}MB를 초과합니다.\n현재 파일: {file_size_mb:.2f}MB\n\n💡 해결 방법:\n- PDF 압축 도구 사용 (예: smallpdf.com)\n- 불필요한 이미지 제거\n- 필요한 페이지만 추출"
        
        # 경고 메시지 (15MB 이상)
        if file_size_mb > 15:
            import streamlit as st
            st.warning(f"⚠️ 파일 크기가 큽니다 ({file_size_mb:.2f}MB). 처리 시간이 오래 걸릴 수 있습니다.")
        
        # BytesIO로 변환 - 파일 포인터를 처음으로 리셋
        uploaded_file.seek(0)
        content = BytesIO(uploaded_file.read())
        content.seek(0)
        
        # 파일이 실제로 PDF인지 확인
        header = content.read(4)
        content.seek(0)
        if header != b'%PDF':
            return None, "❌ 유효한 PDF 파일이 아닙니다. PDF 파일인지 확인해주세요."
        
        return content, None
    except Exception as e:
        return None, f"❌ 파일을 로드할 수 없습니다: {str(e)}"

# ==================== 텍스트 추출 ====================
def extract_text(pdf_file):
    """PDF에서 텍스트를 추출하고 메타데이터를 수집합니다."""
    try:
        # PDF 파일 포인터를 처음으로 리셋
        pdf_file.seek(0)
        
        reader = PdfReader(pdf_file)
        
        # PDF가 비어있는지 확인
        if len(reader.pages) == 0:
            return None, None, "❌ PDF 파일에 페이지가 없습니다."
        
        # 메타데이터 추출
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
        
        # 텍스트 추출
        text = ""
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
            except Exception as e:
                # 개별 페이지 추출 실패는 무시하고 계속 진행
                continue
        
        if not text or len(text.strip()) < 100:
            return None, None, "❌ PDF에서 텍스트를 추출할 수 없습니다. 이미지 기반 PDF이거나 보호된 파일일 수 있습니다."
        
        text = clean_text(text)
        
        return text, metadata, None
    except Exception as e:
        error_msg = str(e)
        if "empty file" in error_msg.lower():
            return None, None, "❌ 빈 파일이거나 손상된 PDF입니다. 다른 파일을 시도해주세요."
        elif "encrypted" in error_msg.lower():
            return None, None, "❌ 암호화된 PDF입니다. 암호를 해제한 후 다시 시도해주세요."
        else:
            return None, None, f"❌ PDF에서 텍스트를 추출할 수 없습니다: {error_msg}"

# ==================== 요약 생성 ====================
def extract_sentences(text):
    """텍스트를 문장 단위로 분리합니다."""
    # 문장 종결 패턴 개선
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z가-힣])', text)
    # 의미있는 문장만 필터링 (최소 30자)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
    return sentences

def summarize(text):
    """텍스트에서 구조화된 요약을 생성합니다."""
    sentences = extract_sentences(text)
    
    if len(sentences) == 0:
        empty_summary = {
            'executive': "추출할 문장이 없습니다.",
            'structured': "요약할 내용이 없습니다.",
            'sections': {},
            'word_count': 0,
            'sentence_count': 0
        }
        return empty_summary
    
    # 기본 통계
    word_count = len(text.split())
    sentence_count = len(sentences)
    
    # 핵심 요약 (처음 5-7 문장)
    executive_length = min(7, max(5, len(sentences) // 10))
    executive = " ".join(sentences[:executive_length])
    
    # 구조화된 요약 (처음 12-15 문장)
    structured_length = min(15, max(12, len(sentences) // 5))
    structured = " ".join(sentences[:structured_length])
    
    # 섹션별 분석 (질적연구방법론 관련)
    sections = identify_sections(text, sentences)
    
    return {
        'executive': executive,
        'structured': structured,
        'sections': sections,
        'word_count': word_count,
        'sentence_count': sentence_count
    }

def identify_sections(text, sentences):
    """텍스트에서 주요 섹션을 식별합니다."""
    sections = {
        '연구 목적 및 배경': {'keywords': ['purpose', 'objective', 'aim', 'goal', 'background', 'introduction', 'context', '목적', '배경', '서론'], 'content': []},
        '이론적 프레임워크': {'keywords': ['theory', 'theoretical', 'framework', 'perspective', 'lens', 'paradigm', '이론', '프레임워크', '관점'], 'content': []},
        '연구 방법': {'keywords': ['method', 'methodology', 'approach', 'design', 'procedure', 'data collection', 'participant', 'sample', '방법', '연구설계', '참여자', '자료수집'], 'content': []},
        '자료 분석': {'keywords': ['analysis', 'coding', 'theme', 'category', 'pattern', 'interpretation', '분석', '코딩', '주제', '범주'], 'content': []},
        '연구 결과': {'keywords': ['result', 'finding', 'outcome', 'emerged', 'revealed', 'discovered', '결과', '발견'], 'content': []},
        '논의 및 함의': {'keywords': ['discussion', 'implication', 'significance', 'contribution', 'limitation', 'future', '논의', '함의', '의의', '한계'], 'content': []}
    }
    
    # 섹션 헤더 탐지
    text_lower = text.lower()
    section_positions = []
    
    for section_name, section_data in sections.items():
        for keyword in section_data['keywords']:
            # 섹션 헤더로 보이는 패턴 찾기
            pattern = rf'\n\s*{re.escape(keyword)}[s]?\s*\n'
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                section_positions.append((match.start(), section_name))
    
    # 위치순 정렬
    section_positions.sort()
    
    # 각 섹션의 내용 추출
    for i, (pos, section_name) in enumerate(section_positions):
        start_pos = pos
        end_pos = section_positions[i + 1][0] if i + 1 < len(section_positions) else len(text)
        
        section_text = text[start_pos:end_pos]
        section_sentences = extract_sentences(section_text)
        
        # 처음 3-5 문장 저장
        sections[section_name]['content'] = section_sentences[:5]
    
    # 헤더를 찾지 못한 경우 키워드 기반 매칭
    for section_name, section_data in sections.items():
        if not section_data['content']:
            for sent in sentences[:100]:  # 처음 100문장만 검사
                sent_lower = sent.lower()
                keyword_count = sum(1 for kw in section_data['keywords'] if kw in sent_lower)
                if keyword_count >= 2:  # 2개 이상의 키워드 매칭
                    idx = sentences.index(sent)
                    section_data['content'] = sentences[idx:min(idx+3, len(sentences))]
                    break
    
    return sections

# ==================== 키워드 추출 ====================
def analyze_keywords(text, top_n=20):
    """TF-IDF와 빈도 분석을 결합하여 키워드를 추출합니다."""
    try:
        # 텍스트 정제
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        
        if len(words) < 20:
            return {'tfidf': [], 'frequency': [], 'academic': []}
        
        # TF-IDF 키워드
        tfidf_keywords = []
        try:
            vectorizer = TfidfVectorizer(
                max_features=top_n,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            tfidf_keywords = sorted(
                zip(feature_names, scores),
                key=lambda x: x[1],
                reverse=True
            )[:top_n]
        except:
            pass
        
        # 빈도 기반 키워드 (불용어 제외)
        word_freq = Counter([w for w in words if w not in STOP_WORDS and len(w) > 4])
        frequency_keywords = word_freq.most_common(top_n)
        
        # 학술 용어 탐지 (질적연구방법론 관련)
        academic_terms = [
            'qualitative', 'quantitative', 'methodology', 'phenomenology',
            'grounded theory', 'case study', 'ethnography', 'narrative',
            'interview', 'observation', 'participant', 'coding', 'theme',
            'category', 'analysis', 'interpretation', 'trustworthiness',
            'credibility', 'transferability', 'dependability', 'confirmability',
            'triangulation', 'saturation', 'reflexivity', 'rigor', 'validity',
            'reliability', 'framework', 'theoretical', 'empirical', 'context'
        ]
        
        found_terms = []
        text_lower = text.lower()
        for term in academic_terms:
            count = text_lower.count(term)
            if count > 0:
                found_terms.append((term, count))
        
        found_terms.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'tfidf': tfidf_keywords,
            'frequency': frequency_keywords,
            'academic': found_terms[:15]
        }
    except Exception as e:
        return {'tfidf': [], 'frequency': [], 'academic': []}

# ==================== 참고문헌 분석 ====================
def analyze_references(text):
    """참고문헌을 추출하고 상세 분석합니다."""
    # References 섹션 찾기
    ref_patterns = [
        r'References\s*\n(.*?)(?=\n\n[A-Z][a-z]+|\Z)',
        r'Bibliography\s*\n(.*?)(?=\n\n[A-Z][a-z]+|\Z)',
        r'References\s*\n(.*)',
        r'REFERENCES\s*\n(.*)',
        r'참고문헌\s*\n(.*)'
    ]
    
    ref_section = ""
    for pattern in ref_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            ref_section = match.group(1)[:10000]  # 처음 10000자
            break
    
    if not ref_section:
        return {
            'items': [],
            'count': 0,
            'years': {},
            'avg_authors': 0,
            'recent_ratio': 0,
            'oldest_year': None,
            'newest_year': None,
            'journal_types': {}
        }
    
    # 참고문헌 항목 추출
    ref_lines = []
    for line in ref_section.split('\n'):
        line = line.strip()
        # 의미있는 참고문헌 라인 (최소 50자, 숫자나 특수문자 포함)
        if len(line) > 50 and re.search(r'[0-9]', line):
            ref_lines.append(line)
    
    # 연도 분석
    years = []
    for line in ref_lines:
        year_matches = re.findall(r'\b(19[5-9]\d|20[0-2]\d)\b', line)
        if year_matches:
            years.extend([int(y) for y in year_matches])
    
    year_dist = Counter(years)
    
    # 최근 논문 비율 (최근 5년)
    from datetime import datetime
    current_year = datetime.now().year
    recent_years = [y for y in years if y >= current_year - 5]
    recent_ratio = (len(recent_years) / len(years) * 100) if years else 0
    
    # 저자 수 분석
    total_authors = 0
    author_counts = []
    
    for line in ref_lines[:50]:  # 처음 50개만 상세 분석
        # 저자 패턴 탐지
        authors = 0
        
        # 패턴 1: "Last, F., Last, F., & Last, F."
        comma_pattern = len(re.findall(r',\s*[A-Z]\.', line))
        authors += comma_pattern
        
        # 패턴 2: "and" 또는 "&"
        and_pattern = len(re.findall(r'\s+(?:and|&)\s+[A-Z]', line, re.IGNORECASE))
        authors += and_pattern
        
        # 패턴 3: "et al."
        if 'et al' in line.lower():
            authors += 3  # et al. 있으면 최소 3명 이상
        
        if authors > 0:
            author_counts.append(authors)
            total_authors += authors
    
    avg_authors = (total_authors / len(author_counts)) if author_counts else 0
    
    # 저널/출판물 유형 분석
    journal_indicators = {
        '저널 논문': ['journal', 'vol.', 'volume', 'pp.', 'pages', 'issue'],
        '학술대회': ['conference', 'proceedings', 'symposium', 'workshop'],
        '단행본': ['book', 'press', 'publisher', 'edition'],
        '학위논문': ['dissertation', 'thesis', 'phd', 'doctoral', 'master']
    }
    
    journal_types = defaultdict(int)
    for line in ref_lines:
        line_lower = line.lower()
        for j_type, indicators in journal_indicators.items():
            if any(indicator in line_lower for indicator in indicators):
                journal_types[j_type] += 1
                break
    
    return {
        'items': ref_lines[:20],  # 상위 20개만 저장
        'count': len(ref_lines),
        'years': dict(year_dist.most_common(15)),
        'avg_authors': round(avg_authors, 1),
        'recent_ratio': round(recent_ratio, 1),
        'oldest_year': min(years) if years else None,
        'newest_year': max(years) if years else None,
        'journal_types': dict(journal_types)
    }

# ==================== 논문 비교 ====================
def compare_papers(papers_data):
    """여러 논문을 체계적으로 비교합니다."""
    if len(papers_data) < 2:
        return None
    
    comparison = {}
    
    # 기본 통계 비교
    comparison['basic_stats'] = []
    for name, data in papers_data.items():
        stats = {
            '논문': name,
            '페이지': data.get('metadata', {}).get('pages', 'N/A'),
            '단어 수': f"{data['summary']['word_count']:,}",
            '문장 수': data['summary']['sentence_count'],
            '참고문헌': data['references']['count']
        }
        comparison['basic_stats'].append(stats)
    
    # 키워드 비교
    all_tfidf = []
    all_academic = []
    
    for name, data in papers_data.items():
        keywords = data.get('keywords', {})
        if keywords.get('tfidf'):
            all_tfidf.extend([kw[0] for kw in keywords['tfidf'][:10]])
        if keywords.get('academic'):
            all_academic.extend([kw[0] for kw in keywords['academic'][:10]])
    
    common_tfidf = [kw for kw, count in Counter(all_tfidf).items() if count > 1]
    common_academic = [kw for kw, count in Counter(all_academic).items() if count > 1]
    
    comparison['common_keywords'] = {
        'tfidf': common_tfidf[:15],
        'academic': common_academic[:15]
    }
    
    # 참고문헌 비교
    ref_comparison = []
    for name, data in papers_data.items():
        refs = data['references']
        ref_stats = {
            '논문': name,
            '참고문헌 수': refs['count'],
            '평균 저자 수': refs['avg_authors'],
            '최근 5년 비율': f"{refs['recent_ratio']}%",
            '연도 범위': f"{refs['oldest_year']}-{refs['newest_year']}" if refs['oldest_year'] else 'N/A'
        }
        ref_comparison.append(ref_stats)
    
    comparison['references'] = ref_comparison
    
    # 연구방법론 용어 비교
    method_terms = ['qualitative', 'quantitative', 'mixed method', 'case study',
                    'grounded theory', 'phenomenology', 'ethnography', 'interview',
                    'survey', 'observation', 'coding', 'theme']
    
    method_presence = {}
    for name, data in papers_data.items():
        text_lower = data['text'].lower()
        found = [term for term in method_terms if term in text_lower]
        method_presence[name] = found
    
    comparison['methodology'] = method_presence
    
    return comparison

# ==================== Streamlit UI ====================
def main():
    st.set_page_config(
        page_title="학술 논문 분석 도구",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 커스텀 CSS
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
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
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
    st.markdown('<div class="sub-header">GPT-4 기반 질적연구방법론 대학원생을 위한 지능형 PDF 분석 시스템</div>', unsafe_allow_html=True)
    
    # API 키 상태 표시
    api_key = load_api_key()
    if api_key:
        st.success("🔑 OpenAI API 키가 설정되었습니다.")
    # 세션 상태 초기화
    if 'papers' not in st.session_state:
        st.session_state.papers = {}
    
    # 사이드바: PDF 업로드
    with st.sidebar:
        st.header("📤 PDF 업로드")
        
        with st.expander("ℹ️ 사용 가이드", expanded=False):
            st.markdown("""
            **📊 기본 분석 기능 (항상 실행):**
            - 구조화된 요약 생성
            - TF-IDF 키워드 분석
            - 참고문헌 심층 분석
            
            **🤖 AI 고급 분석 (선택적 실행):**
            - GPT-4 지능형 요약
            - 질적연구 주제(Theme) 추출
            - 연구질문 및 가설 식별
            - 다중 논문 비교 분석
            
            **📁 파일 크기 권장사항:**
            - 권장: 10MB 이하 (빠른 처리)
            - 최대: 20MB
            - 413 에러 발생 시: PDF 압축 필요
            
            **💡 파일 크기 줄이기:**
            1. smallpdf.com에서 PDF 압축
            2. 불필요한 페이지 제거
            3. 이미지 품질 낮추기
            """)
        
        st.markdown(f"**📊 파일 크기 제한: {MAX_FILE_SIZE_MB}MB**")
        st.caption("⚠️ 파일이 너무 크면 업로드가 실패할 수 있습니다. PDF를 압축하거나 페이지를 줄여보세요.")
        
        uploaded_file = st.file_uploader(
            "PDF 파일을 선택하세요",
            type=['pdf'],
            help=f"학술 논문 PDF 파일 (권장: 10MB 이하, 최대: {MAX_FILE_SIZE_MB}MB)"
        )
        
        paper_name = st.text_input(
            "논문 제목 (선택사항)",
            placeholder="예: Smith et al. (2023) - Qualitative Study",
            help="비워두면 파일명이 사용됩니다"
        )
        
        analyze_button = st.button("🔍 분석 시작", type="primary", use_container_width=True)
        
        if analyze_button:
            if not uploaded_file:
                st.error("❌ PDF 파일을 먼저 업로드해주세요.")
            else:
                # 파일 크기 미리 체크
                file_size_mb = uploaded_file.size / 1024 / 1024
                
                if file_size_mb > MAX_FILE_SIZE_MB:
                    st.error(f"""
                    ❌ 파일이 너무 큽니다!
                    
                    **현재 파일:** {file_size_mb:.2f}MB  
                    **최대 허용:** {MAX_FILE_SIZE_MB}MB
                    
                    **💡 해결 방법:**
                    1. [smallpdf.com](https://smallpdf.com/kr/compress-pdf)에서 PDF 압축
                    2. 불필요한 페이지 제거
                    3. Adobe Acrobat에서 "파일 크기 줄이기" 사용
                    """)
                else:
                    with st.spinner("📄 PDF 처리 중..."):
                        pdf_content, error = load_pdf_from_upload(uploaded_file)
            if not uploaded_file:
                st.error("❌ PDF 파일을 먼저 업로드해주세요.")
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
                                if len(text) < 500:
                                    st.error("❌ 추출된 텍스트가 너무 짧습니다. PDF가 손상되었거나 이미지 기반일 수 있습니다.")
                                else:
                                    # 분석 수행
                                    progress_bar = st.progress(0)
                                    status_text = st.empty()
                                    
                                    status_text.text("📊 기본 분석 중...")
                                    progress_bar.progress(20)
                                    summary = summarize(text)
                                    
                                    status_text.text("🔑 키워드 추출 중...")
                                    progress_bar.progress(50)
                                    keywords = analyze_keywords(text)
                                    
                                    status_text.text("📚 참고문헌 분석 중...")
                                    progress_bar.progress(60)
                                    references = analyze_references(text)
                                    
                                    status_text.text("📊 고급 텍스트 분석 중...")
                                    progress_bar.progress(75)
                                    readability = analyze_readability(text)
                                    complexity = analyze_sentence_complexity(text)
                                    collocations = extract_collocations(text)
                                    discourse = analyze_discourse_markers(text)
                                    citations = extract_citation_patterns(text)
                                    topics_lda = extract_topics_lda(text)
                                    
                                    progress_bar.progress(90)
                                    
                                    # 저장 (GPT 분석은 나중에 선택적으로 수행)
                                    name = paper_name.strip() if paper_name.strip() else uploaded_file.name.replace('.pdf', '')
                                    st.session_state.papers[name] = {
                                        'text': text,
                                        'metadata': metadata,
                                        'summary': summary,
                                        'gpt_summary': None,  # 나중에 생성
                                        'themes': None,
                                        'research_questions': None,
                                        'keywords': keywords,
                                        'references': references,
                                        'readability': readability,
                                        'complexity': complexity,
                                        'collocations': collocations,
                                        'discourse_markers': discourse,
                                        'citation_patterns': citations,
                                        'topics_lda': topics_lda
                                    }
                                    
                                    progress_bar.progress(100)
                                    status_text.text("✅ 분석 완료!")
                                    st.success(f"**'{name}'** 분석이 완료되었습니다!")
                                    st.balloons()
        
        # 로드된 논문 목록
        if st.session_state.papers:
            st.markdown("---")
            st.subheader("📚 로드된 논문")
            
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
    
    # 메인 영역: 결과 표시
    if not st.session_state.papers:
        st.info("👈 **시작하기:** 왼쪽 사이드바에서 PDF 파일을 업로드하고 분석을 시작하세요.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 📝 구조화된 요약")
            st.write("논문의 주요 섹션을 자동으로 식별하고 요약합니다.")
        with col2:
            st.markdown("### 🔑 키워드 분석")
            st.write("TF-IDF와 빈도 분석으로 핵심 키워드를 추출합니다.")
        with col3:
            st.markdown("### 📚 참고문헌 분석")
            st.write("참고문헌의 연도, 저자, 유형을 상세히 분석합니다.")
    
    else:
        # 논문 선택
        selected_paper = st.selectbox(
            "📖 분석할 논문 선택",
            options=list(st.session_state.papers.keys()),
            key="paper_selector"
        )
        
        data = st.session_state.papers[selected_paper]
        
        # 메타데이터 표시
        meta = data['metadata']
        if meta['title'] or meta['author']:
            with st.expander("📋 문서 정보", expanded=False):
                cols = st.columns(4)
                if meta['title']:
                    cols[0].metric("제목", meta['title'][:50] + "...")
                if meta['author']:
                    cols[1].metric("저자", meta['author'][:30] + "...")
                if meta['pages']:
                    cols[2].metric("페이지", meta['pages'])
                if meta['creator']:
                    cols[3].metric("작성 도구", meta['creator'][:30])
        
        # 탭 생성
        tabs = st.tabs([
            "🤖 AI 분석",
            "📊 개요",
            "📈 고급 분석",
            "🎯 주제 & 연구질문",
            "🔑 키워드",
            "📚 참고문헌",
            "🔄 비교 분석"
        ])
        
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = tabs
        
        with tab1:
            st.markdown('<div class="section-header">🤖 GPT-4 기반 지능형 분석</div>', unsafe_allow_html=True)
            
            # GPT 분석 버튼
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info("💡 GPT 분석은 선택적으로 실행할 수 있습니다. 버튼을 눌러 AI 분석을 시작하세요.")
            with col2:
                run_gpt = st.button("🚀 GPT 분석 실행", type="primary", key="gpt_analysis")
            
            if run_gpt:
                with st.spinner("🤖 GPT가 논문을 분석 중입니다... (약 10-20초 소요)"):
                    try:
                        gpt_summary = gpt_summarize(data['text'])
                        themes = gpt_extract_themes(data['text'])
                        research_qs = gpt_research_questions(data['text'])
                        
                        # 세션에 저장
                        st.session_state.papers[selected_paper]['gpt_summary'] = gpt_summary
                        st.session_state.papers[selected_paper]['themes'] = themes
                        st.session_state.papers[selected_paper]['research_questions'] = research_qs
                        
                        st.success("✅ GPT 분석이 완료되었습니다!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ GPT 분석 실패: {str(e)}")
                        st.warning("💡 API 할당량이 부족하거나 네트워크 문제일 수 있습니다. 다른 탭에서 기본 분석 결과를 확인하세요.")
            
            gpt_sum = data.get('gpt_summary', {})
            
            if gpt_sum is None:
                st.warning("⚠️ GPT 분석이 아직 실행되지 않았습니다. 위의 버튼을 눌러 분석을 시작하세요.")
            elif 'error' in gpt_sum:
                st.error(gpt_sum['error'])
                st.info("💡 GPT 분석에 실패했습니다. 다른 탭에서 기본 분석 결과를 확인하세요.")
            else:
                # 핵심 요약
                if '핵심요약' in gpt_sum:
                    st.markdown("#### 📝 핵심 요약")
                    st.info(gpt_sum['핵심요약'])
                
                # 구조화된 섹션
                col1, col2 = st.columns(2)
                
                with col1:
                    if '연구목적' in gpt_sum:
                        st.markdown("#### 🎯 연구 목적")
                        st.write(gpt_sum['연구목적'])
                    
                    if '주요발견' in gpt_sum:
                        st.markdown("#### 🔍 주요 발견")
                        st.write(gpt_sum['주요발견'])
                    
                    if '한계점' in gpt_sum:
                        st.markdown("#### ⚠️ 연구 한계")
                        st.write(gpt_sum['한계점'])
                
                with col2:
                    if '연구방법' in gpt_sum:
                        st.markdown("#### 🔬 연구 방법")
                        st.write(gpt_sum['연구방법'])
                    
                    if '이론적기여' in gpt_sum:
                        st.markdown("#### 💡 이론적 기여")
                        st.write(gpt_sum['이론적기여'])
        
        with tab2:
            st.markdown('<div class="section-header">📊 논문 개요</div>', unsafe_allow_html=True)
            
            # 기본 통계
            col1, col2, col3, col4 = st.columns(4)
            summary = data['summary']
            refs = data['references']
            
            col1.metric("📄 페이지 수", meta['pages'] if meta['pages'] else 'N/A')
            col2.metric("📝 단어 수", f"{summary['word_count']:,}")
            col3.metric("💬 문장 수", summary['sentence_count'])
            col4.metric("📚 참고문헌", refs['count'])
            
            # 핵심 요약 (기본)
            st.markdown('<div class="section-header">핵심 요약</div>', unsafe_allow_html=True)
            st.write(summary['executive'])
            
            # 구조화된 요약
            st.markdown('<div class="section-header">구조화된 요약</div>', unsafe_allow_html=True)
            with st.expander("전체 보기", expanded=False):
                st.write(summary['structured'])
            
            # 상위 키워드 미리보기
            st.markdown('<div class="section-header">주요 키워드 (Top 10)</div>', unsafe_allow_html=True)
            if data['keywords']['tfidf']:
                keywords_preview = [kw[0] for kw in data['keywords']['tfidf'][:10]]
                st.write(" • ".join(keywords_preview))
            else:
                st.info("키워드를 추출할 수 없습니다.")
        
        with tab3:
            st.markdown('<div class="section-header">📈 고급 텍스트 분석</div>', unsafe_allow_html=True)
            st.caption("Python NLP 기술을 활용한 심층 텍스트 분석")
            
            # 가독성 분석
            readability = data.get('readability')
            if readability:
                st.markdown("#### 📖 가독성 분석")
                st.info(f"**난이도:** {readability['difficulty']} (Flesch Reading Ease: {readability['flesch_reading_ease']})")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Flesch-Kincaid Grade", f"{readability['flesch_kincaid_grade']:.1f}")
                col2.metric("SMOG Index", f"{readability['smog_index']:.1f}")
                col3.metric("Coleman-Liau", f"{readability['coleman_liau']:.1f}")
                col4.metric("평균 학년 수준", f"{readability['average_grade_level']:.1f}")
                
                with st.expander("ℹ️ 가독성 지표 설명"):
                    st.markdown("""
                    - **Flesch Reading Ease**: 0-100 점수 (높을수록 읽기 쉬움)
                      - 90-100: 매우 쉬움 (초등 5학년)
                      - 60-70: 표준 (중학교 8-9학년)
                      - 0-30: 매우 어려움 (대학원 수준)
                    - **Grade Level 지표들**: 이해에 필요한 교육 연수
                    - **학술 논문**은 일반적으로 대학(13-16) ~ 대학원(17+) 수준
                    """)
            
            # 문장 복잡도 분석
            complexity = data.get('complexity')
            if complexity:
                st.markdown("#### 📊 문장 복잡도 분석")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("평균 문장 길이", f"{complexity['avg_sentence_length']:.1f} 단어")
                col2.metric("평균 단어 길이", f"{complexity['avg_word_length']:.1f} 글자")
                col3.metric("어휘 다양성 (TTR)", f"{complexity['vocabulary_diversity']:.1f}%")
                
                col4, col5, col6 = st.columns(3)
                col4.metric("총 단어 수", f"{complexity['total_words']:,}")
                col5.metric("고유 단어 수", f"{complexity['unique_words']:,}")
                col6.metric("긴 단어 비율", f"{complexity['long_word_ratio']:.1f}%")
                
                # 문장 길이 분포 시각화
                with st.expander("📈 문장 길이 통계"):
                    st.write(f"**최소 길이:** {complexity['min_sentence_length']} 단어")
                    st.write(f"**최대 길이:** {complexity['max_sentence_length']} 단어")
                    st.write(f"**표준 편차:** {complexity['sentence_length_std']:.2f}")
            
            # Collocation 분석
            collocations = data.get('collocations')
            if collocations:
                st.markdown("#### 🔗 단어 조합 분석 (Collocations)")
                st.caption("통계적으로 유의미하게 함께 나타나는 단어 쌍")
                
                col1, col2 = st.columns(2)
                mid = len(collocations) // 2
                
                with col1:
                    for i, (collocation, freq) in enumerate(collocations[:mid], 1):
                        st.write(f"{i}. **{collocation}** `({freq}회)`")
                
                with col2:
                    for i, (collocation, freq) in enumerate(collocations[mid:], mid+1):
                        st.write(f"{i}. **{collocation}** `({freq}회)`")
            
            # 담화 표지 분석
            discourse = data.get('discourse_markers')
            if discourse:
                st.markdown("#### 💬 담화 표지 분석 (Discourse Markers)")
                st.caption("논증 구조와 논리 전개를 나타내는 언어 표지")
                
                # 바 차트로 시각화
                discourse_df = pd.DataFrame([
                    {'카테고리': k, '빈도': v} for k, v in discourse.items()
                ])
                
                fig = px.bar(discourse_df, x='빈도', y='카테고리', 
                           orientation='h',
                           title='담화 표지 사용 빈도',
                           color='빈도',
                           color_continuous_scale='blues')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("ℹ️ 담화 표지 설명"):
                    st.markdown("""
                    - **인과관계**: 원인과 결과를 연결 (because, therefore, thus 등)
                    - **대조**: 상반된 아이디어 제시 (however, but, although 등)
                    - **추가**: 정보 추가 (furthermore, moreover, also 등)
                    - **예시**: 구체적 예시 제공 (for example, such as 등)
                    - **결론**: 논지 마무리 (in conclusion, to sum up 등)
                    - **강조**: 주장 강화 (indeed, in fact, clearly 등)
                    """)
            
            # LDA 토픽 모델링
            topics_lda = data.get('topics_lda')
            if topics_lda:
                st.markdown("#### 🏷️ 토픽 모델링 (LDA)")
                st.caption("잠재 디리클레 할당 기법으로 추출한 주요 토픽")
                
                for topic in topics_lda:
                    with st.expander(f"**토픽 {topic['topic_id']}**", expanded=False):
                        st.write("**주요 단어:**")
                        words_with_scores = [f"{word} ({score:.3f})" 
                                           for word, score in zip(topic['words'][:5], topic['scores'][:5])]
                        st.write(" • ".join(words_with_scores))
                        
                        st.write("\n**전체 단어:**")
                        st.write(", ".join(topic['words']))
            
            # 인용 패턴 분석
            citations = data.get('citation_patterns')
            if citations:
                st.markdown("#### 📝 인용 패턴 분석")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("(Author, Year)", citations.get('author_year', 0))
                col2.metric("(Author, Year, p.X)", citations.get('author_year_page', 0))
                col3.metric("[숫자]", citations.get('numbered', 0))
                col4.metric("et al. 사용", citations.get('multiple_authors', 0))
                
                total_citations = sum(citations.values())
                if total_citations > 0:
                    st.info(f"📊 **총 인용 횟수:** {total_citations}회")
                    
                    # 인용 스타일 비율
                    citation_df = pd.DataFrame([
                        {'스타일': 'Author-Year', '빈도': citations.get('author_year', 0)},
                        {'스타일': 'Author-Year-Page', '빈도': citations.get('author_year_page', 0)},
                        {'스타일': 'Numbered', '빈도': citations.get('numbered', 0)}
                    ])
                    
                    fig = px.pie(citation_df, values='빈도', names='스타일',
                               title='인용 스타일 분포')
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.markdown('<div class="section-header">🎯 주제 & 연구질문</div>', unsafe_allow_html=True)
            
            # 연구질문
            rqs = data.get('research_questions')
            
            if rqs is None:
                st.info("💡 'AI 분석' 탭에서 GPT 분석을 실행하면 연구질문과 주제를 자동으로 추출합니다.")
            elif 'error' not in rqs:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### ❓ 연구질문")
                    if rqs.get('연구질문'):
                        for i, rq in enumerate(rqs['연구질문'], 1):
                            st.write(f"**RQ{i}:** {rq}")
                    else:
                        st.info("연구질문을 찾지 못했습니다.")
                
                with col2:
                    st.markdown("#### 💭 연구가설")
                    if rqs.get('연구가설'):
                        for i, h in enumerate(rqs['연구가설'], 1):
                            st.write(f"**H{i}:** {h}")
                    else:
                        st.info("연구가설을 찾지 못했습니다.")
            
            # 주제 분석
            themes = data.get('themes')
            
            if themes is not None and 'error' not in themes:
                st.markdown("#### 🏷️ 주요 주제 (Themes)")
                if themes.get('주요주제'):
                    cols = st.columns(3)
                    for i, theme in enumerate(themes['주요주제']):
                        cols[i % 3].info(f"**주제 {i+1}**\n\n{theme}")
                else:
                    st.info("주제를 추출하지 못했습니다.")
                
                st.markdown("#### 🧩 핵심 개념")
                if themes.get('핵심개념'):
                    concept_text = " • ".join(themes['핵심개념'])
                    st.write(concept_text)
                else:
                    st.info("핵심 개념을 추출하지 못했습니다.")
            
            # 섹션별 분석 (기본)
            st.markdown('<div class="section-header">섹션별 분석</div>', unsafe_allow_html=True)
            
            sections = summary['sections']
            selected_sections = ['연구 목적 및 배경', '연구 방법', '연구 결과', '논의 및 함의']
            
            for section_name in selected_sections:
                if section_name in sections:
                    section_data = sections[section_name]
                    with st.expander(f"**{section_name}**", expanded=False):
                        if section_data['content']:
                            for sent in section_data['content'][:3]:
                                st.write(f"• {sent}")
                        else:
                            st.info("이 섹션의 내용을 식별하지 못했습니다.")
        
        with tab5:
            st.markdown('<div class="section-header">🔑 키워드 분석</div>', unsafe_allow_html=True)
            
            keywords = data['keywords']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 TF-IDF 키워드")
                st.caption("문서 내 중요도 기반 키워드")
                
                if keywords['tfidf']:
                    for i, (keyword, score) in enumerate(keywords['tfidf'][:15], 1):
                        st.write(f"{i}. **{keyword}** `{score:.4f}`")
                else:
                    st.info("TF-IDF 키워드를 추출할 수 없습니다.")
            
            with col2:
                st.markdown("#### 🔢 빈도 기반 키워드")
                st.caption("출현 빈도 기반 키워드")
                
                if keywords['frequency']:
                    for i, (keyword, count) in enumerate(keywords['frequency'][:15], 1):
                        st.write(f"{i}. **{keyword}** `{count}회`")
                else:
                    st.info("빈도 키워드를 추출할 수 없습니다.")
            
            # 학술 용어
            st.markdown("#### 🎓 학술 및 방법론 용어")
            st.caption("질적/양적 연구방법론 관련 용어")
            
            if keywords['academic']:
                # 3열로 표시
                cols = st.columns(3)
                for i, (term, count) in enumerate(keywords['academic']):
                    col_idx = i % 3
                    cols[col_idx].write(f"**{term}** ({count})")
            else:
                st.info("학술 용어를 찾지 못했습니다.")
        
        with tab6:
            st.markdown('<div class="section-header">📚 참고문헌 분석</div>', unsafe_allow_html=True)
            
            refs = data['references']
            
            if refs['count'] == 0:
                st.warning("참고문헌 섹션을 찾을 수 없습니다.")
            else:
                # 통계 요약
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("📚 총 참고문헌", refs['count'])
                col2.metric("👥 평균 저자 수", refs['avg_authors'])
                col3.metric("📅 최근 5년 비율", f"{refs['recent_ratio']}%")
                
                if refs['oldest_year'] and refs['newest_year']:
                    year_range = f"{refs['oldest_year']}-{refs['newest_year']}"
                    col4.metric("📆 연도 범위", year_range)
                
                # 연도별 분포
                st.markdown("#### 📅 연도별 참고문헌 분포")
                if refs['years']:
                    st.bar_chart(refs['years'])
                else:
                    st.info("연도 정보를 추출할 수 없습니다.")
                
                # 출판물 유형
                if refs['journal_types']:
                    st.markdown("#### 📖 출판물 유형 분포")
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.bar_chart(refs['journal_types'])
                    with col2:
                        for j_type, count in refs['journal_types'].items():
                            percentage = (count / refs['count'] * 100)
                            st.write(f"**{j_type}**: {count}개 ({percentage:.1f}%)")
                
                # 참고문헌 목록
                st.markdown("#### 📋 참고문헌 목록 (상위 20개)")
                with st.expander("전체 목록 보기", expanded=False):
                    for i, ref in enumerate(refs['items'], 1):
                        st.write(f"{i}. {ref}")
        
        with tab7:
            st.markdown('<div class="section-header">🔄 논문 비교 분석</div>', unsafe_allow_html=True)
            
            if len(st.session_state.papers) < 2:
                st.info("💡 비교 분석을 위해서는 최소 2개의 논문을 업로드해주세요.")
                st.markdown("""
                **🤖 AI 기반 비교 분석 기능:**
                - GPT-4를 활용한 지능형 논문 비교
                - 공통 주제 및 차별점 식별
                - 연구방법론 비교
                - 기본 통계 비교
                - 키워드 및 참고문헌 패턴 분석
                """)
            else:
                # GPT 기반 비교
                st.markdown("#### 🤖 AI 기반 심층 비교")
                
                if st.button("🚀 GPT 비교 분석 실행", type="primary", key="gpt_compare"):
                    with st.spinner("🤖 GPT가 논문들을 비교 분석 중입니다... (약 20-30초 소요)"):
                        try:
                            paper_texts = {name: data['text'] for name, data in st.session_state.papers.items()}
                            gpt_comp = gpt_compare_papers(paper_texts)
                            
                            if 'error' not in gpt_comp:
                                st.success("✅ AI 비교 분석 완료!")
                                
                                if '공통주제' in gpt_comp:
                                    st.markdown("##### 🎯 공통 주제")
                                    for theme in gpt_comp['공통주제']:
                                        st.write(f"• {theme}")
                                
                                if '차별점' in gpt_comp:
                                    st.markdown("##### 🔍 주요 차별점")
                                    st.info(gpt_comp['차별점'])
                                
                                if '방법론비교' in gpt_comp:
                                    st.markdown("##### 🔬 방법론 비교")
                                    st.write(gpt_comp['방법론비교'])
                                
                                if '종합평가' in gpt_comp:
                                    st.markdown("##### 📊 종합 평가")
                                    st.success(gpt_comp['종합평가'])
                            else:
                                st.error(gpt_comp['error'])
                                st.warning("💡 API 할당량 문제일 수 있습니다. 아래 기본 비교 결과를 확인하세요.")
                        except Exception as e:
                            st.error(f"❌ GPT 비교 분석 실패: {str(e)}")
                            st.warning("💡 아래 기본 비교 결과를 확인하세요.")
                
                st.markdown("---")
                
                # 기본 통계 비교
                comparison = compare_papers(st.session_state.papers)
                
                if comparison:
                    st.markdown("#### 📊 기본 통계 비교")
                    st.table(comparison['basic_stats'])
                    
                    # 키워드 비교
                    st.markdown("#### 🔑 공통 키워드")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**TF-IDF 공통 키워드:**")
                        if comparison['common_keywords']['tfidf']:
                            st.write(" • ".join(comparison['common_keywords']['tfidf']))
                        else:
                            st.info("공통 키워드 없음")
                    
                    with col2:
                        st.markdown("**학술 용어 공통 키워드:**")
                        if comparison['common_keywords']['academic']:
                            st.write(" • ".join(comparison['common_keywords']['academic']))
                        else:
                            st.info("공통 학술 용어 없음")
                    
                    # 참고문헌 비교
                    st.markdown("#### 📚 참고문헌 비교")
                    st.table(comparison['references'])
                    
                    # 연구방법론 비교
                    st.markdown("#### 🔬 연구방법론 용어 비교")
                    for paper_name, terms in comparison['methodology'].items():
                        with st.expander(f"**{paper_name}**"):
                            if terms:
                                st.write(" • ".join(terms))
                            else:
                                st.info("방법론 관련 용어를 찾지 못했습니다.")

if __name__ == "__main__":
    main()
