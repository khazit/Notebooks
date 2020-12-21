"""Slightly modified version of :
https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py
"""


import argparse
from os.path import join, isdir

import cv2
import numpy as np

import gym
from gym.envs.box2d import car_racing


car_racing.STATE_H = 150
car_racing.STATE_W = 150

parser = argparse.ArgumentParser()
parser.add_argument("--delta", default=100)
parser.add_argument("--out_dir", default=None)


if __name__ == "__main__":
    from pyglet.window import key

    a = np.array([0.0, 0.0, 0.0])
    args = parser.parse_args()
    assert isdir(args.out_dir), "Out directory not found."

    def key_press(k, mod):
        global restart
        if k == 0xFF0D:
            restart = True
        if k == key.LEFT:
            a[0] = -1.0
        if k == key.RIGHT:
            a[0] = +1.0
        if k == key.UP:
            a[1] = +1.0
        if k == key.DOWN:
            a[2] = +0.8  # set 1.0 for wheels to block to zero rotation

    def key_release(k, mod):
        if k == key.LEFT and a[0] == -1.0:
            a[0] = 0
        if k == key.RIGHT and a[0] == +1.0:
            a[0] = 0
        if k == key.UP:
            a[1] = 0
        if k == key.DOWN:
            a[2] = 0

    env = car_racing.CarRacing()
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    isopen = True
    rounds = 0
    while isopen:
        env.reset()
        rounds += 1
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            observations, reward, done, info = env.step(a)
            total_reward += reward
            if steps % int(args.delta) == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print(
                    "round {} step {} total_reward {:+0.2f}".format(rounds, steps, total_reward)
                )
                cv2.imwrite(
                    join(args.out_dir, f"round_{rounds}_step_{steps}.bmp"),
                    observations[:128, 11:-11, [2, 1, 0]]
                )
            steps += 1
            isopen = env.render()
            if done or restart or isopen == False:
                break
    env.close()
