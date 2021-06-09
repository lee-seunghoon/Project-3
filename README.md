# 자녀 목소리 AI 애착인형 프로젝트


## 배경

> - 수명증가, 독거노인 증가, 고독사 등 `노인 세대`의 사회적 문제가 커지고 있다.
> - `코로나19`로 인해 활동이 제한되며 정서적 외로움, 우울증이 증가하고 있다.
> - 경제적, 신체적, 정서적 측면의 다양한 어려움이 있지만, `정서적 측면`에 집중하기로 결정



## 벤치마킹

> - 국내 : 효돌이 효순이
> - 국외 : 러봇
> - AI 기술을 인형에 접목하여, 알림, 대화 등 다양한 기능을 통해 어르신의 정서적 외로움을 해소

![image-20210609110143504](md-images/image-20210609110143504.png)

## 차별점

> - 자녀 목소리 합성 기능
> - 어르신 질문 데이터 기반 감정 분석
> - 어르신 감정 상태 인터페이스 제공



## 프로젝트 일정 및 기간

> - 전체기간 : `2021.04.28 ~ 2021.06.04` (약 1개월)
>   - `04.28 ~ 05.03` : 프로젝트 주제 선정 및 배경 탐색
>   - `05.05 ~ 05.25` : 각 파트 개발
>   - `05.17 ~ 05.18` : 중간점검 피드백
>   - `05.22` : 각 파트 기능 통합 완료
>   - `05.24 ~ 06.03` : 발표 준비 및 기능 업데이트 
>   - `2021.06.04` : 최종 발표 



## 기술 스택

> - Tensorflow2
> - Google Colab
> - AWS EC2
> - Konlpy
> - Scikit Learn
> - Numpy & Pandas



## 담당 역할

> - 말벗서비스 - 대화 기능 개발
> - PPT 제작



## 문제점

> - 어르신 맞춤 대화 데이터셋 구축 어려움
> - 어르신 대화 데이터 부족으로 딥러닝 대화 모델 성능 저하
> - 어르신 특성상 사투리 인식 어려움



## 해결 방안

> - AI hub 일상 대화, 건강 대화 데이터 활용하려 2,000개 이상 대화 데이터 확보
> - TF-IDF 와 코사인 유사도를 활용하여 예약어 기반 대화 모델 개발
> - 사투리 대화 데이터 자체제작하여 보강 (약 100개) 



## 배운점

> - 예약어 기반 댸화 프로세스 이해 및 개발 가능
> - RNN, LSTM, Seqeunce2Sequence, BERT 등 자연어 처리 관련 딥러닝 모델
> - AI 모델을 서비스로 실행하는 과정 이해
> - MQTT 프로토콜 숙지



## 수상

> - 개인 노력상 (총 60여명 중 10명 수상)

![멀티캠퍼스 노력상](md-images/%EB%A9%80%ED%8B%B0%EC%BA%A0%ED%8D%BC%EC%8A%A4%20%EB%85%B8%EB%A0%A5%EC%83%81.jpg)

## 코드리뷰

