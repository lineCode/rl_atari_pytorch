# Atari Pytorch

> A atari AI Player implement by pytorch play games

## Synopsis

Reinforcement learning shows the most potential of AI in many area, however, to use reinforcement learning you must specific your environment which is somethings hard to build a environment for your problem. But gym let us has a very convenient way to explore rl algorithms.

So here it is, using DDPG and LSTM to play atari, **and it is really effective!!**, as I can show in Pong-V0-moniter you can find the play progress in mp4. Our AI can really beat computer!!

## How to Play With

OK, to play with it, simply run:
```
./run_train.sh
```
This will train on Pong-V0 env, and save your model into `checkpoints/`. If you interrupted, next time it will continue train on last saved model.

And, to play with your model, simply run:
```
./run_play.sh
```

You can change env in .sh command, many atari env are supported.

## Future

This is a very good exploration but not the end, later on I will explore on reinforcement learning on autonamous-car driving problem and train a AI to fucking drive!!

## Contribute

Well, very welcome to send PR to add more game env train models to this repo!!
If you have any question about this you can find me via wechat: `jintianiloveu`.
