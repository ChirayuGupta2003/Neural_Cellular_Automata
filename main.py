import pygame as pg
import numpy as np
from numba import jit, cuda, njit, prange
import time
import random


pg.init()
clock = pg.time.Clock()
font = pg.font.Font(None, 36)

HEIGHT = 1000
WIDTH = 1000

CELL_SIDE = 1

grid_cols = WIDTH // CELL_SIDE
grid_rows = HEIGHT // CELL_SIDE


screen = pg.display.set_mode((WIDTH, HEIGHT))


@jit(nopython=True, parallel=True)
def initialize():
    # img = cv.imread("./img.jpg")
    # game_matrix = cv.resize(img, (grid_cols, grid_rows))
    # print(game_matrix)
    # print(game_matrix.shape)
    game_matrix = np.random.randint(0, 2, (grid_rows, grid_cols))
    # game_matrix = np.zeros((grid_rows, grid_cols, channels))
    return game_matrix


cols = random.random(), random.random(), random.random()


@jit(nopython=True, parallel=True)
def draw_screen_backend(game_matrix):
    screen_array = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    for row in prange(grid_rows):
        for col in prange(grid_cols):
            x, y, w, h = col * CELL_SIDE, row * CELL_SIDE, CELL_SIDE, CELL_SIDE
            for i in prange(3):
                screen_array[x:x+w, y:y+h,
                             i] = game_matrix[row, col] * 255 * cols[i]
    return screen_array


def draw_screen(screen_: pg.Surface, game_matrix):
    screen_array = draw_screen_backend(game_matrix)
    pg.surfarray.blit_array(screen_, screen_array)
    display_fps()


@jit(nopython=True, parallel=True)
def inverted_gaussian(x):
    return -1/(2**(0.6*x**2)) + 1


@jit(nopython=True, parallel=True)
def game_of_life_activation(x):
    if x == 3. or x == 11. or x == 12.:
        return 1.
    return 0.


def identity_activation(x):
    return x


@jit(nopython=True, parallel=True)
def gaussian(x, b):
    return 1/(2**((x-b)**2))


@jit(nopython=True, parallel=True)
def pathways_activation(x):
    return gaussian(x, 3.5)


@jit(nopython=True, parallel=True)
def waves_activation(x):
    return abs(1.2*x)


@jit(nopython=True, parallel=True)
def slime_mould_activation(x):
    result = -1. / (0.89 * x**2 + 1.) + 1.
    return result


@jit(nopython=True, parallel=True)
def main(game_matrix: np.ndarray):

    worm_filter = np.array(
        [[0.68, -.9, 0.68], [-.9, -.66, -.9], [0.68, -.9, 0.68]])
    game_of_life_filter = np.array(
        [
            [1, 1, 1],
            [1, 9, 1],
            [1, 1, 1]
        ]
    )

    random_filter = np.array(
        [
            [-0.448, 0.256, -0.448],
            [0.256, 0.246, 0.256],
            [-0.448, 0.256, -0.448],
        ]
    )

    pathways_filter = np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ]
    )

    slime_mould_filter = np.array(
        [
            [.8, -.85, .8],
            [-.85, -.2, -.85],
            [.8, -.85, .8]
        ]
    )

    waves_filter = np.array(
        [
            [0.5645999908447266, -.7159000039100647, .5645999908447266],
            [-.7159000039100647, .6269000172615051, -.7159000039100647],
            [0.5645999908447266, -.7159000039100647, .5645999908447266],
        ]
    )

    mitosis_filter = np.array(
        [
            [-0.9390000104904175, 0.8799999952316284, -0.9390000104904175],
            [0.8799999952316284,   0.4000000059604645, 0.8799999952316284],
            [-0.9390000104904175, 0.8799999952316284, -0.9390000104904175],
        ]
    )

    rows, cols = game_matrix.shape

    temp_game_matrix = np.zeros(game_matrix.shape)

    for row in prange(rows):
        for col in prange(cols):
            ans = 0
            for i in prange(-1, 2):
                for j in prange(-1, 2):
                    x_val = col + i
                    y_val = row + j

                    if x_val < 0:
                        x_val = cols + x_val
                    elif x_val >= cols:
                        x_val = x_val - cols

                    if y_val < 0:
                        y_val = rows + y_val
                    elif y_val >= rows:
                        y_val = y_val - rows

                    ans += game_matrix[y_val, x_val] * \
                        slime_mould_filter[i+1, j+1]

            temp_game_matrix[row, col] = slime_mould_activation(ans)

    return temp_game_matrix


def display_fps():
    fps = str(int(clock.get_fps()))
    fps_text = font.render(
        f"FPS: {fps} IPS: {float(fps) * iterations_per_frame}", True, (255, 255, 255))
    screen.blit(fps_text, (10, 10))


grid_changed = True
iterations_per_frame = 1
running = True
gameMatrix = initialize()


while running:

    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE:
                running = False
            if event.key == pg.K_r:
                gameMatrix = initialize()
            if event.key == pg.K_RIGHT:
                iterations_per_frame += 1
                print(iterations_per_frame)
            if event.key == pg.K_LEFT:
                iterations_per_frame -= 1
                iterations_per_frame = max(1, iterations_per_frame)
                print(iterations_per_frame)

    if grid_changed:
        draw_screen(screen, gameMatrix)
        grid_changed = False

    if pg.mouse.get_pressed()[0]:
        mouse_pos = pg.mouse.get_pos()
        x, y = mouse_pos
        row, col = x // CELL_SIDE, y // CELL_SIDE
        slice_rows = slice(row - 5, row + 5)
        slice_cols = slice(col - 5, col + 5)
        gameMatrix[slice_cols, slice_rows] = 1
        grid_changed = True

    for i in range(iterations_per_frame):
        gameMatrix = main(gameMatrix)
    grid_changed = True

    pg.display.flip()
    clock.tick(144)


pg.quit()
