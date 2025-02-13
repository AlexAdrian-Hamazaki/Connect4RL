# Connect4RL

This repo holds code for a multi-agent RL model for playing connect4 that is trained with curriculum learning

### About training the model...

I trained this model on the Connect4 PettingZoo out of the box environment (found [!here](https://pettingzoo.farama.org/environments/classic/connect_four/))

I trained it using a DQN developed in AgileRL (found [!here](https://docs.agilerl.com/en/latest/api/algorithms/dqn.html#dqn))

The first curriculum I trained the model on (can be found as lesson1.yml). Here I train the model against a randomly-picking opponent (10,000 games)

In the second curriculum (can be found as lesson2.yml). I trained the model to play against itself (50,000 games)


### The model playing Connect 4

Here's an example of the current iteration of the agent (red) beating a randomly picking agent (black)

![random](/videos/lesson2/connect_four_random_opp.gif)

And an example of the agent playing against itself

![self](/videos/lesson2/connect_four_self_opp.gif)


### Insights gained from the model

Currently I'm gathering insights from the model

Connect 4 is technically a ![solved game](https://en.wikipedia.org/wiki/Connect_Four) (see "Mathemetical solution). If both players play perfectly,
player 1 should begin by placing their tile in the middle-column, and they will win on the 41rst turn.

To explore what the model has learned, I played 1,000,000 games with the model, and created a PowerBI dashboard to explore what the model has learned.



Insight 1: Win rate of player 1
- win rate if first tile is in middle column


Insight 2: The best first tile
- Win rates according to where you put the first tile

Insight 3: Expected game length with optimal play
- Expected game length
