transformer계열을 사용하여 챗봇을 학습을 수행하였을때 정답 시퀀스를 맞추지 못하고
일반화 목적만 달성하는데 반해 generic망으로 구성한 챗봇은 정답 시퀀스를 98%이상 
맞추고 일반화도 달성한다.

실행방법
1. chatbot_data.zip, dual_chatbot;0.zip 압축 해제
2. python dual_chatbot.py 를 실행하여 학습 결과 추론 수행
3. dual_chatbot.py의 no_train을 막고 train을 활성시켜 학습 수행, 이때 dual_chatbot;0.dfx를 삭제하여 처음부터 학습
