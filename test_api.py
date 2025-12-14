#!/usr/bin/env python3
"""OpenAI API 연결 테스트"""

import os
import sys
from pathlib import Path
from openai import OpenAI

# secrets.toml에서 API 키 읽기
secrets_file = Path(__file__).parent / ".streamlit" / "secrets.toml"

api_key = None

# 1. 환경변수 확인
api_key = os.environ.get('OPENAI_API_KEY')
if api_key:
    print(f"✓ 환경변수에서 API 키 발견: {api_key[:15]}...")
else:
    print("✗ 환경변수에 API 키 없음")

# 2. secrets.toml 파일 확인
if secrets_file.exists():
    print(f"\n✓ secrets.toml 파일 존재: {secrets_file}")
    with open(secrets_file, 'r') as f:
        content = f.read()
        print(f"파일 내용:\n{content[:200]}")
        
        # TOML 파싱
        import re
        match = re.search(r'openai_api_key\s*=\s*["\']([^"\']+)["\']', content)
        if match:
            api_key = match.group(1)
            print(f"\n✓ secrets.toml에서 API 키 발견: {api_key[:15]}...")
        else:
            print("\n✗ secrets.toml에 openai_api_key 없음")
else:
    print(f"\n✗ secrets.toml 파일 없음: {secrets_file}")

# 3. config/api_keys.json 확인
config_file = Path(__file__).parent / "config" / "api_keys.json"
if config_file.exists():
    print(f"\n✓ config/api_keys.json 존재")
    import json
    with open(config_file, 'r') as f:
        config = json.load(f)
        if 'openai_api_key' in config:
            api_key = config['openai_api_key']
            print(f"✓ config에서 API 키 발견: {api_key[:15]}...")
        else:
            print("✗ config에 openai_api_key 없음")
else:
    print(f"\n✗ config/api_keys.json 파일 없음")

# API 테스트
if not api_key:
    print("\n❌ API 키를 찾을 수 없습니다!")
    sys.exit(1)

print(f"\n{'='*60}")
print("OpenAI API 연결 테스트 시작")
print(f"{'='*60}")

try:
    client = OpenAI(api_key=api_key)
    print("✓ OpenAI 클라이언트 초기화 성공")
    
    # 간단한 테스트 요청
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "안녕하세요. 간단히 '테스트 성공'이라고만 답변해주세요."}
        ],
        max_tokens=50
    )
    
    result = response.choices[0].message.content
    print(f"\n✓ API 호출 성공!")
    print(f"응답: {result}")
    print(f"\n{'='*60}")
    print("✅ 모든 테스트 통과!")
    print(f"{'='*60}")
    
except Exception as e:
    print(f"\n❌ API 호출 실패: {e}")
    print(f"\n오류 타입: {type(e).__name__}")
    print(f"{'='*60}")
    sys.exit(1)
