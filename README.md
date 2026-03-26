# LLM 기반 강의 영상 요약 및 질의응답 시스템

> 2271521 박재휘 | 자연어처리 기말 프로젝트

---

## 1. 배경 및 필요성

- **미디어 콘텐츠 폭증** (유튜브, 온라인 강의, 회의 등) → 긴 영상에서 핵심 내용을 찾기 어려움
- **정보 탐색 비용 증가** → 사용자는 영상 전체를 시청하지 않고도 빠르게 질문하고 답을 원함
- **해결책: 텍스트 기반 질의응답 시스템**
  - 자막을 벡터로 인덱싱하여 LLM을 통해 자연어 질문에 응답
  - 요약 기능 포함 → 영상 내용을 텍스트 수준에서 압축 가능

---

## 2. 목표

**자연어로 질의 시, 관련된 정확한 응답을 제공하는 시스템 개발**

- 다양한 미디어 콘텐츠에서 정보를 추출
- LLM을 결합한 지능형 정보 접근
- 미디어 콘텐츠에 기반한 질의응답 시스템 개발

---

## 3. 시스템 개요

```
[영상, 오디오]
      ↓
[Faster-Whisper 자막 생성]  ← STT (Speech To Text)
      ↓
[LlamaIndex + LLM]          ← 문서 임베딩 + 벡터 인덱싱
      ↓
[사용자 질문]
      ↓
[자연어 요약/응답 출력]
```

| 단계 | 설명 |
|------|------|
| 입력 | 영상, 오디오 |
| 처리 | 음성 인식(STT), 자막 추출 |
| 질의 | 사용자 질문 입력 |
| 응답 | LLM을 통한 자연어 응답 제공 |

---

## 4. 기술 구성

### 4-1. STT: Faster-Whisper

- **모델 선택**: Large-v3 vs Medium 비교 → 성능 유사하나 **Medium이 속도가 빠름** → Medium 채택
- **처리 성능**: 6분 37초 영상을 35정도만에 처리
- 한국어 영상은 `task="transcribe"`, 외국어는 `task="translate"` (영어로 번역, 한국어→한국어 직접 지원)

```python
from faster_whisper import WhisperModel
model = WhisperModel("medium", compute_type="float16")

if detected_lang == "ko":
    segments, _ = model.transcribe(video_file, beam_size=5, task="transcribe", language="ko")
else:
    segments, _ = model.transcribe(video_file, beam_size=5, task="translate", language=detected_lang)
```

### 4-2. LlamaIndex + LLM

- 추출된 자막 텍스트를 저장 → **문서 임베딩 + 벡터 인덱싱** → LLM을 통해 질문에 대한 요약 또는 응답 생성
- 환경: 코랩 무료 GPU → 무거운 모델 사용 불가

**모델 비교 및 선택**

| 모델 | 비고 |
|------|------|
| `Llama-3.2-1B-Instruct` | 영어 기반, 한국어 품질 낮음 |
| `HyperCLOVAX-SEED-Text-Instruct-1.5B` | 한국어 영상에서 더 좋은 성능 → **최종 선택** |

- 임베딩 모델: `sentence-transformers/all-MiniLM-L6-v2`
- 청크 설정: `chunk_size=512`, `chunk_overlap=50`
- 질의 예시: `"Please summarize the key points of this video in Korean."`

### 4-3. 실행 결과

**토론 영상 (다화자, 다양한 어조)**
- 북한/미국 관련 공약 질문 → 문서 기반 적절한 대답 생성
- 후보 목록 질문 → `'~~~후보'` 텍스트가 문서에 존재하나 제대로 인식 못하는 한계 발생
<img width="1561" height="606" alt="image" src="https://github.com/user-attachments/assets/0f68c82d-4cbd-4838-a7c9-8f42885084de" />

<img width="1585" height="594" alt="image" src="https://github.com/user-attachments/assets/045f34e9-6fa9-47d0-b4e4-8dc15f42c64c" />


**강의 영상 (단일 화자)**
- 인공신경망, 기계학습 주요 개념 질의 → 정확한 응답
- `debug` 명령으로 참조한 문서 청크 및 유사도 점수 확인 가능
<img width="1157" height="556" alt="image" src="https://github.com/user-attachments/assets/f7fc275f-c4bd-4a3a-b303-ecd69b238af6" />

<img width="1583" height="513" alt="image" src="https://github.com/user-attachments/assets/3701c7af-3767-40f5-9280-d10b93d8c1ae" />



**사용 가능한 명령어**

| 명령어 | 기능 |
|--------|------|
| 질문 입력 | 자연어 질의응답 |
| `debug` | 마지막 답변의 소스 확인 |
| `history` | 대화 기록 보기 |
| `summary` | 강의 요약 다시 보기 |
| `check_doc` | 문서 내용 일부 확인 |
| `exit` | 종료 |

---

## 5. 한계점 및 개선 방안

| 한계점 | 원인 | 개선 방안 |
|--------|------|-----------|
| **정확도 제한** | LLM이 문서에 없는 이름/내용 생성 가능 (환각) | 문서 chunking 최적화 |
| **단순 질의응답 형태** | 단일 질문-응답 방식, 대화 흐름/사용자 의도 파악 없음 | LangChain Memory 도입 |
| **초기 요약 품질 저하** | `similarity_top_k`로 가져온 청크가 요약에 부적절할 경우 공허하거나 무의미 | 요약 전용 query_engine 별도 구성, 특정 스코어 이상 노드만 선택하는 조건 추가 |
| **CLI 기반 인터페이스 불편** | 실제 사용자들이 사용하기에 직관적이지 않음 | 웹 프론트엔드 연동, 버튼 기반 요약/소스 보기/질문창 구현 |

---
