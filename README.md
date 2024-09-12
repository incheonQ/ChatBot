# LangChain, Streamlit, Chatbot 실습

이 프로젝트는 Streamlit을 사용하여 만든 고객 응대 챗봇입니다. 
PDF 파일을 임베딩하고 OpenAI의 GPT 모델을 사용하여 질문에 답변합니다.

## 주요 기능

- PDF 파일 업로드 및 임베딩
- OpenAI GPT 모델을 사용한 질문 응답
- 대화 내용 및 임베딩 초기화 기능

## 사용된 기술

- Streamlit
- LangChain
- OpenAI GPT
- FAISS (벡터 데이터베이스)

## 설치 방법

1. 저장소를 클론합니다:
   ```
   git clone [저장소 URL]
   ```

2. 필요한 패키지를 설치합니다:
   ```
   pip install -r requirements.txt
   ```

3. `.env` 파일을 생성하고 OpenAI API 키를 추가합니다:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## 사용 방법

1. Streamlit 앱을 실행합니다:
   ```
   streamlit run chatbot.py
   ```

2. 웹 브라우저에서 앱에 접속합니다.

3. 사이드바에서 모델과 PDF 파일을 선택합니다.

4. 채팅 입력창에 질문을 입력하고 응답을 받습니다.

## 주의사항

- PDF 파일은 `./files` 디렉토리에 저장해야 합니다.
- OpenAI API 키가 필요합니다.


## References

 - [실습자료 Notion 1](https://cobalt-clock-cf4.notion.site/1-316cddb045e1499e85b6e318d7e53016)

- [실습자료 Notion 2](https://cobalt-clock-cf4.notion.site/2-d7701792858d4f029a257d80fbadc588)

- [실습자료GitHub](https://github.com/Ukbang/240828_Modulabs/tree/main?tab=readme-ov-file)
