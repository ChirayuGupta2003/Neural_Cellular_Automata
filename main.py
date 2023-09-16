import pygame as pg
import numpy as np
from numba import jit, cuda, njit
import time
import random
import pygame_gui
from pygame_gui.elements import UITextEntryLine
from pygame_gui.elements.ui_drop_down_menu import UIDropDownMenu
from pygame_gui.elements.ui_button import UIButton


pg.init()
clock = pg.time.Clock()
font = pg.font.Font(None, 36)

HEIGHT = 1000
WIDTH = 1000

CELL_SIDE = 1

grid_cols = WIDTH // CELL_SIDE
grid_rows = HEIGHT // CELL_SIDE
screen = pg.display.set_mode((WIDTH, HEIGHT))




def initialize():
    game_matrix = np.random.randint(0, 2, (grid_rows, grid_cols))
    return game_matrix


cols = random.random(), random.random(), random.random()


@njit
def draw_screen_backend(game_matrix):
    screen_array = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    for row in range(grid_rows):
        for col in range(grid_cols):
            x, y, w, h = col * CELL_SIDE, row * CELL_SIDE, CELL_SIDE, CELL_SIDE
            for i in range(3):
                screen_array[x:x+w, y:y+h,
                             i] = game_matrix[row, col] * 255 * cols[i]
    return screen_array


def draw_screen(screen_: pg.Surface, game_matrix):
    screen_array = draw_screen_backend(game_matrix)
    pg.surfarray.blit_array(screen_, screen_array)
    display_fps()


@njit
def inverted_gaussian(x):
    return -1/(2**(0.6*x**2)) + 1


@njit
def game_of_life_activation(x):
    if x == 3. or x == 11. or x == 12.:
        return 1.
    return 0.


def identity_activation(x):
    return x


@njit
def gaussian(x, b):
    return 1/(2**((x-b)**2))


@njit
def pathways_activation(x):
    return gaussian(x, 3.5)


@njit
def waves_activation(x):
    return abs(1.2*x)


@njit
def slime_mould_activation(x):
    result = -1. / (0.89 * x**2 + 1.) + 1.
    return result


@njit(parallel=True, target_backend=cuda)
def worm(game_matrix: np.ndarray):

    worm_filter = np.array(
        [
             [0.68, -.9, 0.68],
             [-.9, -.66, -.9],
             [0.68, -.9, 0.68]
         ]
    )
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

    for row in range(rows):
        for col in range(cols):
            ans = 0
            for i in range(-1, 2):
                for j in range(-1, 2):
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

manager = pygame_gui.UIManager((HEIGHT,WIDTH))

# Create a simple menu element
filter_options = ["Choose filter to use","Worm","Game of Life","Random","Pathways","Slime Mould","Waves","Mitosis"]
drop_down1 = UIDropDownMenu(
    options_list=filter_options,
    starting_option="Choose filter to use",
    relative_rect = pg.Rect((10,200),(200,30)),
    manager=manager

)

# Create UITextEntryLine objects and associated increment/decrement buttons
small_font = pg.font.Font(None, 18)
number_input1 = UITextEntryLine(
    relative_rect=pg.Rect(10, 40, 40, 30),
    manager=manager,
)
number_input1.set_allowed_characters('numbers')


number_input2 = UITextEntryLine(
    relative_rect=pg.Rect(60, 40, 40, 30),
    manager=manager,
    # set_allowed_characters = 'numbers'
)
number_input2.set_allowed_characters('numbers')


number_input3 = UITextEntryLine(
    relative_rect=pg.Rect(110, 40, 40, 30),
    manager=manager,
    # set_allowed_characters = 'numbers'
)
number_input3.set_allowed_characters('numbers')


number_input4 = UITextEntryLine(
    relative_rect=pg.Rect(10, 80, 40, 30),
    manager=manager,
    # set_allowed_characters = 'numbers'
)
number_input4.set_allowed_characters('numbers')


number_input5 = UITextEntryLine(
    relative_rect=pg.Rect(60, 80, 40, 30),
    manager=manager,
    # set_allowed_characters = 'numbers'
)
number_input5.set_allowed_characters('numbers')


number_input6 = UITextEntryLine(
    relative_rect=pg.Rect(110, 80, 40, 30),
    manager=manager,
    # set_allowed_characters = 'numbers'
)
number_input6.set_allowed_characters('real')


number_input7 = UITextEntryLine(
    relative_rect=pg.Rect(10, 120, 40, 30),
    manager=manager,
    # set_allowed_characters = 'numbers'
)
number_input7.set_allowed_characters('numbers')


number_input8 = UITextEntryLine(
    relative_rect=pg.Rect(60, 120, 40, 30),
    manager=manager,
    # set_allowed_characters = 'numbers'
)
number_input8.set_allowed_characters('numbers')

number_input9 = UITextEntryLine(
    relative_rect=pg.Rect(110, 120, 40, 30),
    manager=manager,
    # set_allowed_characters = 'numbers'
)
number_input9.set_allowed_characters('numbers')

current_filter = worm

manager.draw_ui(screen)
# Update the screen
pg.display.flip()

while running:
    manager.draw_ui(screen)
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
        if event.type == pg.USEREVENT:
            if event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
                if event.ui_element == drop_down1:
                    selected_option = event.text
                    if selected_option == "Choose filter to use":
                        current_filter = worm
                    if selected_option == "Worm":
                        current_filter = worm
                    if selected_option == "Game of life":
                        current_filter = game_of_life_activation
                    if selected_option == "Random":
                        current_filter = random
                    if selected_option == "Pathways":
                        current_filter = pathways_activation
                    if selected_option == "Slime Mould":
                        current_filter = slime_mould_activation
                    if selected_option == "Waves":
                        current_filter = waves_activation
                    if selected_option == "Mitosis":
                        current_filter = waves_activation
        # manager.process_events(event)
    if current_filter is not None:
        gameMatrix = current_filter(gameMatrix)
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
        gameMatrix = slime_mould_activation(gameMatrix)
    grid_changed = True

    manager.update(clock.tick(60) / 1000.0)

    # Draw the UI elements
    manager.draw_ui(screen)

    pg.display.flip()
    clock.tick(144)


pg.quit()


