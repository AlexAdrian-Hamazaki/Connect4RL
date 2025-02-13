import os

import imageio
import numpy as np
import torch
from CurriculumEnv import *
from pettingzoo.classic import connect_four_v3
from PIL import Image, ImageDraw, ImageFont

from agilerl.algorithms.dqn import DQN


# Define function to return image
def _label_with_episode_number(frame, episode_num, frame_no, p):
    im = Image.fromarray(frame)
    drawer = ImageDraw.Draw(im)
    text_color = (255, 255, 255)
    font = ImageFont.truetype("/usr/share/fonts/opentype/urw-base35/NimbusRoman-Italic.otf", size=45)
    
    drawer.text(
        (100, 5),
        f"Episode: {episode_num+1}     Frame: {frame_no}",
        fill=text_color,
        font=font,
    )
    if p == 1:
        player = "Player 1"
        color = (255, 0, 0)
    if p == 2:
        player = "Player 2"
        color = (100, 255, 150)
    if p is None:
        player = "Self-play"
        color = (255, 255, 255)
        
    # Get text bounding box to compute width
    text = f"Agent: {player}"
    bbox = drawer.textbbox((0, 0), text, font=font)  # Get (x0, y0, x1, y1)
    text_width = bbox[2] - bbox[0]

    # Calculate x-position for right-aligned text
    image_width = im.width
    x_pos = image_width - text_width - 20  # 20px padding from the right edge

    # Draw Agent Label
    drawer.text((x_pos, 5), text, fill=color, font=font)
    
    return im


# Resizes frames to make file size smaller
def resize_frames(frames, fraction):
    resized_frames = []
    for img in frames:
        new_width = int(img.width * fraction)
        new_height = int(img.height * fraction)
        img_resized = img.resize((new_width, new_height))
        resized_frames.append(np.array(img_resized))

    return resized_frames


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lesson = 2
    path = f"models/DQN/lesson{lesson}_trained_agent.pt"  # Path to saved agent checkpoint

    env = connect_four_v3.env(render_mode="rgb_array")
    env.reset()

    # Configure the algo input arguments
    state_dim = [
        env.observation_space(agent)["observation"].shape for agent in env.agents
    ]
    one_hot = False
    action_dim = [env.action_space(agent).n for agent in env.agents]

    # Pre-process dimensions for pytorch layers
    # We will use self-play, so we only need to worry about the state dim of a single agent
    # We flatten the 6x7x2 observation as input to the agent's neural network
    state_dim = np.zeros(state_dim[0]).flatten().shape
    action_dim = action_dim[0]
    
    
            # Set the seed for reproducibility
    seed = 42  # Change as needed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load the saved agent
    dqn = DQN.load(path, device)
    


    for opponent_difficulty in ["random", "self"]:
        # Create opponent
        if opponent_difficulty == "self":
            opponent = dqn
        else:
            opponent = Opponent(env, opponent_difficulty)

        # Define test loop parameters
        episodes = 5  # Number of episodes to test agent on
        max_steps = (
            500  # Max number of steps to take in the environment in each episode
        )

        rewards = []  # List to collect total episodic reward
        frames = []  # List to collect frames

        print("============================================")
        print(f"Agent: {path}")
        print(f"Opponent: {opponent_difficulty}")

        # Test loop for inference
        for ep in range(episodes):
            opponent_first = False
            p = 1

            if opponent_difficulty == "self":
                p = None
            env.reset()  # Reset environment at start of episode
            frame = env.render()
            frames.append(
                _label_with_episode_number(frame, episode_num=ep, frame_no=0, p=p)
            )
            observation, reward, done, truncation, _ = env.last()
            player = -1  # Tracker for which player's turn it is
            score = 0
            for idx_step in range(max_steps):
                action_mask = observation["action_mask"]
                if player < 0:
                    state, _ = transform_and_flip(observation, player=0)
                    if opponent_first:
                        if opponent_difficulty == "self":
                            action = opponent.get_action(
                                state, epsilon=0.1, action_mask=action_mask
                            )[0]
                        elif opponent_difficulty == "random":
                            action = opponent.get_action(action_mask)
                        # else:
                        #     action = opponent.get_action(player=0)
                    else:
                        action = dqn.get_action(
                            state, epsilon=0, action_mask=action_mask
                        )[
                            0
                        ]  # Get next action from agent
                if player > 0:
                    state, _ = transform_and_flip(observation, player=1)
                    if not opponent_first:
                        if opponent_difficulty == "self":
                            action = opponent.get_action(
                                state, epsilon=0.1, action_mask=action_mask
                            )[0]
                        elif opponent_difficulty == "random":
                            action = opponent.get_action(action_mask)
                        # else:
                        #     action = opponent.get_action(player=1)
                    else:
                        action = dqn.get_action(
                            state, epsilon=0, action_mask=action_mask
                        )[
                            0
                        ]  # Get next action from agent
                env.step(action)  # Act in environment
                observation, reward, termination, truncation, _ = env.last()
                reward = -reward
                # Save the frame for this step and append to frames list
                frame = env.render()
                frame = _label_with_episode_number(
                        frame, episode_num=ep, frame_no=idx_step, p=p
                    )
        
                frames.append(frame)

                # Will be true agent if on agent's turn
                if (player > 0 and opponent_first) or (
                    player < 0 and not opponent_first # player is -1, opponent went second, so opponent is player 1
                ):
                    # reward = env.reward(done=termination, player=1)

                    score += reward


                # Stop episode if any agents have terminated
                if truncation or termination:
                    break

                player *= -1

            print("-" * 15, f"Episode: {ep+1}", "-" * 15)
            print(f"Episode length: {idx_step}")
            print(f"Score: {score}")
            # Add 5 copies of the same frame
            frames.extend([frame] * 5)
            

        print("============================================")

        frames = resize_frames(frames, 0.5)

        # Save the gif to specified path
        gif_path = f"./videos/lesson{lesson}"
        os.makedirs(gif_path, exist_ok=True)
        imageio.mimwrite(
            os.path.join(gif_path, f"connect_four_{opponent_difficulty}_opp.gif"),
            frames,
            duration=400,
            loop=True,
        )

    env.close()