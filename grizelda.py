from typing import Dict, Tuple, Optional
import numpy as np
import pyglet
from pyglet import gl
from pyglet.window import Window


class ImageViewer:
    def __init__(self):
        self.window = None
        self.isopen = False

    def __del__(self):
        self.close()

    def imshow(self, img: np.ndarray, caption: Optional[str] = None):
        height, width, _ = img.shape
        pitch = -3 * width
        
        if self.window is None:
            self.window = Window(width=width, height=height, vsync=False)
            self.width = width
            self.height = height
            self.isopen = True

        data = img.tobytes()
        image = pyglet.image.ImageData(width, height, 'RGB', data, pitch=pitch)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)

        texture = image.get_texture()
        texture.width = self.width
        texture.height = self.height

        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        texture.blit(0, 0)
        self.window.flip()

        if caption is not None:
            self.window.set_caption(caption)

    def close(self):
        if self.isopen:
            self.window.close()
            self.window = None
            self.isopen = False


class RecurrentAgent:
    def __init__(self, seed: int, hidden_size: int = 16):
        self.seed = seed
        self.np_random = np.random.RandomState(seed=seed)

        self.input_size = 3  # (agent position, time)
        self.output_size = 4  # (up, down, left, right)
        self.hidden_size = hidden_size

        self.t = np.zeros((1, 1))
        self.delta_t = self.np_random.rand()

        # recurrent network
        self.weight1 = self.np_random.randn(self.input_size + self.hidden_size, self.hidden_size)
        self.bias1 = self.np_random.randn(1, self.hidden_size)
        self.init_h = self.np_random.randn(1, self.hidden_size)

        self.weight2 = self.np_random.randn(self.hidden_size, self.output_size)
        self.bias2 = self.np_random.randn(1, self.output_size)

        self.reset()

    def reset(self):
        self.t.fill(0)
        self.hidden = np.array(self.init_h)

    def __call__(self, inputs: np.ndarray, deterministic: bool = True) -> int:
        inputs = np.concatenate([inputs, self.hidden, np.cos(self.t)], axis=1)
        self.hidden = np.tanh(np.matmul(inputs, self.weight1) + self.bias1)
        output = np.matmul(self.hidden, self.weight2) + self.bias2
        e_x = np.exp(output - np.max(output))
        p = e_x / e_x.sum()
        if deterministic:
            action = np.argmax(p, axis=1)
        else:
            action = self.np_random.choice(self.output_size, p=p)
        self.t += self.delta_t
        return action


class Grizelda:
    def __init__(self, size: int = 9, deterministic: bool = True):
        self.seed()

        self.size = size
        self.deterministic = deterministic

        self.grid = np.zeros((size, size), dtype=np.bool)
        self.grid[0, :] = 1
        self.grid[:, 0] = 1
        self.grid[-1, :] = 1
        self.grid[:, -1] = 1

        self._enemy1 = RecurrentAgent(self.np_random.randint(2 ** 31 - 1))
        self._enemy2 = RecurrentAgent(self.np_random.randint(2 ** 31 - 1))

        self.reset()

        self.viewer = ImageViewer()

    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed=seed)

    @property
    def observation(self) -> np.ndarray:
        agent_layer = np.zeros((self.size, self.size), dtype=np.bool)
        agent_layer[self._agent_pos] = 1

        enemy_layer = np.zeros((self.size, self.size), dtype=np.bool)
        if not self._enemy1_dead:
            enemy_layer[self._enemy1_pos] = 1
        if not self._enemy2_dead:
            enemy_layer[self._enemy2_pos] = 1

        key_layer = np.zeros((self.size, self.size), dtype=np.bool)
        if self.key_visible:
            key_layer[self._key_pos] = 1
        
        door_layer = np.zeros((self.size, self.size), dtype=np.bool)
        door_layer[self._door_pos] = 1

        grid_layer = np.array(self.grid)
        if self._door_opened:
            grid_layer[self._door_pos] = 0

        obs = np.stack([
            agent_layer,
            enemy_layer,
            key_layer,
            door_layer,
            grid_layer,
        ], axis=0)
        return obs

    @property
    def info(self) -> Dict[str, np.ndarray]:
        return None

    def reset(self) -> np.ndarray:
        self._agent_pos = self._random_pos()
        self._agent_dir = 0  # up
        self._attack_pos = None

        self._enemy1.reset()
        self._enemy1_pos = self._random_pos()
        self._enemy1_dead = False

        self._enemy2.reset()
        self._enemy2_pos = self._random_pos()
        self._enemy2_dead = False
        
        self._key_pos = self._random_pos()
        self._key_picked_up = False
        
        self._door_pos = self._random_door_pos()
        self._door_opened = False

        return self.observation

    def _random_pos(self) -> Tuple[int, int]:
        i, j = self.np_random.randint(1, self.size - 1, size=2)
        while self.grid[i, j] == 1:
            i, j = self.np_random.randint(1, self.size - 1, size=2)
        return i, j

    def _random_door_pos(self) -> Tuple[int, int]:
        return [
            (0, self.size // 2),              # top
            (self.size - 1, self.size // 2),  # bottom
            (self.size // 2, 0),              # left
            (self.size // 2, self.size - 1),  # right
        ][self.np_random.choice(4)]

    def _normalize_pos(self, pos: Tuple[int, int]) -> Tuple[float, float]: 
        i, j = pos
        i = i / self.size - 0.5
        j = j / self.size - 0.5
        return (i, j)

    @property
    def agent_pos(self) -> Tuple[float, float]:
        return self._normalize_pos(self._agent_pos)

    @property
    def enemy1_pos(self) -> Tuple[float, float]:
        if self._enemy1_dead:
            return (-1.0, -1.0)
        return self._normalize_pos(self._enemy1_pos)

    @property
    def enemy2_pos(self) -> Tuple[float, float]:
        if self._enemy2_dead:
            return (-1.0, -1.0)
        return self._normalize_pos(self._enemy2_pos)

    @property
    def key_visible(self) -> bool:
        return not self._key_picked_up and self._enemy1_dead and self._enemy2_dead

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        agent_i, agent_j = self._agent_pos
        self._attack_pos = None

        if action == 0:  # move up
            self._agent_dir = 0
            agent_i -= 1
        elif action == 1:  # move down
            self._agent_dir = 1
            agent_i += 1
        elif action == 2:  # move left
            self._agent_dir = 2
            agent_j -= 1
        elif action == 3:  # move right
            self._agent_dir = 3
            agent_j += 1
        elif action == 4:  # attack
            self._attack_pos = [
                (agent_i - 1, agent_j),  # attack up
                (agent_i + 1, agent_j),  # attack down
                (agent_i, agent_j - 1),  # attack left
                (agent_i, agent_j + 1),  # attack right
            ][self._agent_dir]

        # the agent escaped the lab, game over!
        if not(0 <= agent_i < self.size) or not(0 <= agent_j < self.size):
            return self.observation, +1.0, True, self.info

        # the agent died, game over!
        if (agent_i, agent_j) in [self._enemy1_pos, self._enemy2_pos]:
            return self.observation, -1.0, True, self.info

        if self.grid[agent_i, agent_j] == 0:
            self._agent_pos = (agent_i, agent_j)

        if (agent_i, agent_j) == self._door_pos and self._key_picked_up:
            self.open_door()

        if (agent_i, agent_j) == self._key_pos and self.key_visible:
            self.pick_up_key()

        if self._attack_pos is not None:
            self.attack()

        # move the enemies if they're not dead
        if not self._enemy1_dead:
            enemy1_i, enemy1_j = self._enemy1_pos
            enemy1_action = self._enemy1(np.array([self.agent_pos]),
                                         deterministic=self.deterministic)

            if enemy1_action == 0:  # up
                enemy1_i -= 1
            elif enemy1_action == 1:  # down
                enemy1_i += 1
            elif enemy1_action == 2:  # left
                enemy1_j -= 1
            elif enemy1_action == 3:  # right
                enemy1_j += 1
            else:
                raise ValueError(f'invalid action: {enemy1_action}')

            if self.grid[enemy1_i, enemy1_j] == 0:
                self._enemy1_pos = (enemy1_i, enemy1_j)

            # the agent died, game over!
            if self._enemy1_pos == self._agent_pos:
                return self.observation, -1.0, True, self.info

        if not self._enemy2_dead:
            enemy2_i, enemy2_j = self._enemy2_pos
            enemy2_action = self._enemy2(np.array([self.agent_pos]), 
                                         deterministic=self.deterministic)

            if enemy2_action == 0:  # up
                enemy2_i -= 1
            elif enemy2_action == 1:  # down
                enemy2_i += 1
            elif enemy2_action == 2:  # left
                enemy2_j -= 1
            elif enemy2_action == 3:  # right
                enemy2_j += 1
            else:
                raise ValueError(f'invalid action: {enemy2_action}')

            if self.grid[enemy2_i, enemy2_j] == 0:
                self._enemy2_pos = (enemy2_i, enemy2_j)

            # the agent died, game over!
            if self._enemy2_pos == self._agent_pos:
                return self.observation, -1.0, True, self.info

        return self.observation, 0.0, False, self.info

    def pick_up_key(self):
        self._key_pos = None
        self._key_picked_up = True

    def open_door(self):
        self.grid[self._door_pos] = 0
        self._door_opened = True

    def attack(self):
        if self._attack_pos == self._enemy1_pos:
            self._enemy1_dead = True
            self._enemy1_pos = None
        if self._attack_pos == self._enemy2_pos:
            self._enemy2_dead = True
            self._enemy2_pos = None

    def render(self, mode: str = 'human') -> np.ndarray:
        canvas = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) == self._agent_pos:
                    canvas[i, j, :] = (0, 116, 217)  # blue
                elif (i, j) == self._attack_pos:
                    canvas[i, j, :] = (255, 65, 54)  # red
                elif (i, j) == self._enemy1_pos:
                    canvas[i, j, :] = (240, 18, 190)  # magenta
                elif (i, j) == self._enemy2_pos:
                    canvas[i, j, :] = (255, 133, 27)  # orange
                elif (i, j) == self._key_pos and self.key_visible:
                    canvas[i, j, :] = (46, 204, 64)  # green
                elif (i, j) == self._door_pos:
                    if self._door_opened:
                        canvas[i, j, :] = (255, 255, 255)  # white
                    else:
                        canvas[i, j, :] = (64, 64, 64)  # Grey
                elif self.grid[i, j]:  # wall
                    canvas[i, j, :] = (0, 0, 0)  # black
                else:
                    canvas[i, j, :] = (255, 255, 255)  # white

        if mode == 'rgb_array':
            return canvas
        elif mode == 'human':
            canvas = np.kron(canvas, np.ones((8, 8, 1))).astype(np.uint8)
            self.viewer.imshow(canvas)
            return canvas
        else:
            raise NotImplementedError

    def close(self):
        self.viewer.close()


class MetaZelda(Grizelda):
    def step(self, action):
        obs, reward, done, info = super().step(action)
        if done:
            obs = self.reset()
        return obs, reward, info


if __name__ == '__main__':
    from time import sleep
    from pyglet.window import key

    env = MetaZelda()

    env.render(mode='human')  # initialize viewer
    
    def key2action(viewer):
        key_state_handler = key.KeyStateHandler()
        viewer.window.push_handlers(key_state_handler)
        while True:
            if key_state_handler[key.UP]:
                yield 0
            elif key_state_handler[key.DOWN]:
                yield 1
            elif key_state_handler[key.LEFT]:
                yield 2
            elif key_state_handler[key.RIGHT]:
                yield 3
            elif key_state_handler[key.SPACE]:
                yield 4
            yield -1

    try:
        for action in key2action(env.viewer):
            env.render()
            env.step(action)
            sleep(1e-1)

    except KeyboardInterrupt:
        pass

    env.close()
    