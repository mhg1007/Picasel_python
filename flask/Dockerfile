# 베이스 이미지로 Python 3.10을 사용
FROM python:3.10

# 필수 시스템 패키지 설치 (libGL)
RUN apt-get update && \
    apt-get install -y libgl1 && \
    rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 필요 패키지 설치를 위한 requirements 파일 복사
COPY requirements.txt .

# Flask와 TensorFlow 설치
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

# Flask 앱을 실행하기 위해 환경 변수 설정 (필요한 경우 수정)
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Flask 포트 설정
EXPOSE 5000

# Flask 앱 실행
CMD ["flask", "run", "--host=0.0.0.0"]
