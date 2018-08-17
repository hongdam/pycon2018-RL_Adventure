# Pycon2018 "RL Adventure : DQN 부터 Rainbow DQN까지"
참조 reference : https://github.com/higgsfield/RL-Adventure

## 발표자료
그림을 클릭하시면 자료화면으로 넘어갑니다

[<img src="https://image.slidesharecdn.com/pyconkr2018rladventure-180817113545/95/pycon2018-rl-adventure-dqn-rainbow-dqn-1-638.jpg?cb=1534507317" title="SlideShare자료화면으로 넘어갑니다">](https://www.slideshare.net/ssuserbd7730/pyconkr-2018-rladventureto-the-rainbow)

## 발표내용
- 파트 1 : DQN, Double & Dueling DQN - 성태경 
- 파트 2 : PER and NoisyNet - 양홍선 
- 파트 3 : Distributed RL - 이의령 
- 파트 4 : RAINBOW - 김예찬 


## 실험환경
**openai gym** : atari
- CartPole-v1
- PongNoFrameskip-v4


## 구현체
- DQN
- DDQN
- Dueling DQN
- PER
- Noisy DQN
- Categorical DQN
- Rainbow


## 실험결과 확인
<table>
  <tr align="center">
    <td>CartPole-v1</td>
    <td>PongNoFrameskip-v4</td>
  </tr>
  <tr align="center">
    <td>
      <img src="https://github.com/hongdam/pycon2018-RL_Adventure/blob/master/code/save_gif/CartPole-v1_RainbowDQN_10.gif?raw=true" width="315" height="210"/>
    </td>
    <td>
      <img src="https://github.com/hongdam/pycon2018-RL_Adventure/blob/master/code/save_gif/PongNoFrameskip-v4_RainbowDQN_3.gif?raw=true" width="160" height="210"/>
    </td>
  </tr>
</table>
