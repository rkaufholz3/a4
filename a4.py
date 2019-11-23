import mdptoolbox, mdptoolbox.example
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd


def generate_aima_grid():

    # https://stats.stackexchange.com/questions/339592/how-to-get-p-and-r-values-for-a-markov-decision-process-grid-world-problem

    """
    AIMA example.  The manual way. 12 states. (0, 0) is top left, (4, 3) is bottom right
    Actions are Up (0), Down (1), Left (2), Right (3) in every state.
    P 0.8 of going in intended direction.  P 0.2 of going in right angle to intended action.
    Collision with wall results in no movement.

    # Transition model: Prob of reaching state s' if action a is done from state s
    # P shape: (4, 12, 12) : 4 actions x 12 from states x 12 to states.
    """

    actions_map = {0: '^', 1: 'V', 2: '<', 3: '>'}

    transitions = np.zeros((4, 12, 12))  # (A, S, S)
    transitions[0] = [[0.9, 0.1, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0.1, 0.8, 0.1, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0.1, 0.8, 0.1, 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0.8, 0., 0., 0., 0.2, 0., 0., 0., 0., 0., 0., 0.],
             [0., 0.8, 0., 0., 0.1, 0., 0.1, 0., 0., 0., 0., 0.],
             [0., 0., 0.8, 0., 0., 0., 0.1, 0.1, 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.8, 0., 0., 0., 0.1, 0.1, 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0.1, 0.8, 0.1, 0.],
             [0., 0., 0., 0., 0., 0., 0.8, 0., 0., 0.1, 0., 0.1],
             [0., 0., 0., 0., 0., 0., 0., 0.8, 0., 0., 0.1, 0.1]]

    transitions[1] = [[0.1, 0.1, 0., 0., 0.8, 0., 0., 0., 0., 0., 0., 0.],
             [0.1, 0.8, 0.1, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0.1, 0., 0.1, 0., 0., 0.8, 0., 0., 0., 0., 0.],
             [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.2, 0., 0., 0., 0.8, 0., 0., 0.],
             [0., 0., 0., 0., 0.1, 0., 0.1, 0., 0., 0.8, 0., 0.],
             [0., 0., 0., 0., 0., 0., 0.1, 0.1, 0., 0., 0.8, 0.],
             [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0.9, 0.1, 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0.1, 0.8, 0.1, 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.1, 0.8, 0.1],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.1, 0.9]]

    transitions[2] = [[0.9, 0., 0., 0., 0.1, 0., 0., 0., 0., 0., 0., 0.],
             [0.8, 0.2, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0.8, 0.1, 0., 0., 0., 0.1, 0., 0., 0., 0., 0.],
             [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0.1, 0., 0., 0., 0.8, 0., 0., 0., 0.1, 0., 0., 0.],
             [0., 0.1, 0., 0., 0.8, 0., 0., 0., 0., 0.1, 0., 0.],
             [0., 0., 0.1, 0., 0., 0., 0.8, 0., 0., 0., 0.1, 0.],
             [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.1, 0., 0., 0., 0.9, 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0.8, 0.2, 0., 0.],
             [0., 0., 0., 0., 0., 0., 0.1, 0., 0., 0.8, 0.1, 0.],
             [0., 0., 0., 0., 0., 0., 0., 0.1, 0., 0., 0.8, 0.1]]

    transitions[3] = [[0.1, 0.8, 0., 0., 0.1, 0., 0., 0., 0., 0., 0., 0.],
             [0., 0.2, 0.8, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0.1, 0.8, 0., 0., 0.1, 0., 0., 0., 0., 0.],
             [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0.1, 0., 0., 0., 0.8, 0., 0., 0., 0.1, 0., 0., 0.],
             [0., 0.1, 0., 0., 0., 0., 0.8, 0., 0., 0.1, 0., 0.],
             [0., 0., 0.1, 0., 0., 0., 0., 0.8, 0., 0., 0.1, 0.],
             [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0.1, 0., 0., 0., 0.1, 0.8, 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.2, 0.8, 0.],
             [0., 0., 0., 0., 0., 0., 0.1, 0., 0., 0., 0.1, 0.8],
             [0., 0., 0., 0., 0., 0., 0., 0.1, 0., 0., 0., 0.9]]

    rewards = np.asarray((-0.02, -0.02, -0.02, 1, -0.02, -0.02, -0.02, -1, -0.02, -0.02, -0.02, -0.02))

    print('\n***** aima grid world *****\n')
    print('Transition matrix:', transitions.shape)
    print('Reward matrix:', rewards.shape)

    return transitions, rewards


def generate_grid_world():

    # https://stats.stackexchange.com/questions/339592/how-to-get-p-and-r-values-for-a-markov-decision-process-grid-world-problem

    # # 4 by 3 world
    # grid_size = (3, 4)  # (3, 4)
    # black_cells = [(1, 1)]  # [(1, 1)]
    # white_cell_reward = -0.04  # -0.02
    # green_cell_loc = (0, 3)  # (0, 3)  the goal (terminate)
    # red_cell_loc = (1, 3)  # (1, 3)  penalty cell (terminate)
    # green_cell_reward = 1.0  # 1.0
    # red_cell_reward = -1.0  # -1.0
    # action_lrfb_prob = (.1, .1, .8, 0.)  # (.1, .1, .8, 0.)
    # start_loc = (0, 0)

    # # 10 by 10 world
    # grid_size = (10, 10)  # (3, 4)
    # black_cells = [(1, 1)]  # [(1, 1)]
    # white_cell_reward = -0.2  # -0.02
    # green_cell_loc = (0, 3)  # (0, 3)  the goal (terminate)
    # red_cell_loc = (1, 3)  # (1, 3)  penalty cell (terminate)
    # green_cell_reward = 1.0  # 1.0
    # red_cell_reward = -1.0  # -1.0
    # action_lrfb_prob = (.1, .1, .8, 0.)  # (.1, .1, .8, 0.)
    # start_loc = (0, 0)

    # 50 by 50 world
    # grid_size = (100, 100)  # (3, 4)
    # black_cells = [(10, 10), (10, 11), (10, 12), (11, 10), (11, 11), (11, 12)]  # [(1, 1)]
    # white_cell_reward = -0.02  # -0.02
    # green_cell_loc = (9, 9)  # (0, 3)  the goal (terminate)
    # red_cell_loc = (9, 10)  # (1, 3)  penalty cell (terminate)
    # green_cell_reward = 10000.0  # 1.0
    # red_cell_reward = -10000.0  # -1.0
    # action_lrfb_prob = (.1, .1, .8, 0.)  # (.1, .1, .8, 0.)
    # start_loc = (0, 0)

    # 80 by 60 world
    grid_size = (60, 80)  # (3, 4)
    black_cells = [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)]  # [(1, 1)]
    white_cell_reward = -0.02  # -0.02
    green_cell_loc = (30, 40)  # (0, 3)  the goal (terminate)
    red_cell_loc = (31, 40)  # (1, 3)  penalty cell (terminate)
    green_cell_reward = 1000.0  # 1.0
    red_cell_reward = -1.0  # -1.0
    action_lrfb_prob = (.1, .1, .8, 0.)  # (.1, .1, .8, 0.)
    start_loc = (0, 0)

    # green_cell is the goal
    # red_cell is the penalty cell

    num_states = grid_size[0] * grid_size[1]
    num_actions = 4
    P = np.zeros((num_actions, num_states, num_states))
    R = np.zeros((num_states, num_actions))

    # helpers
    to_2d = lambda x: np.unravel_index(x, grid_size)
    to_1d = lambda x: np.ravel_multi_index(x, grid_size)

    def hit_wall(cell):
        if cell in black_cells:
            return True
        try:  # ...good enough...
            to_1d(cell)
        except ValueError as e:
            return True
        return False

    # make probs for each action
    a_up = [action_lrfb_prob[i] for i in (0, 1, 2, 3)]
    a_down = [action_lrfb_prob[i] for i in (1, 0, 3, 2)]
    a_left = [action_lrfb_prob[i] for i in (2, 3, 1, 0)]
    a_right = [action_lrfb_prob[i] for i in (3, 2, 0, 1)]
    actions = [a_up, a_down, a_left, a_right]
    for i, a in enumerate(actions):
        actions[i] = {'up': a[2], 'down': a[3], 'left': a[0], 'right': a[1]}

    # work in terms of the 2d grid representation

    def update_P_and_R(cell, new_cell, a_index, a_prob):

        if cell == green_cell_loc:
            P[a_index, to_1d(cell), to_1d(cell)] = 1.0
            R[to_1d(cell), a_index] = green_cell_reward

        elif cell == red_cell_loc:
            P[a_index, to_1d(cell), to_1d(cell)] = 1.0
            R[to_1d(cell), a_index] = red_cell_reward

        elif hit_wall(new_cell):  # add prob to current cell
            P[a_index, to_1d(cell), to_1d(cell)] += a_prob
            R[to_1d(cell), a_index] = white_cell_reward

        else:
            P[a_index, to_1d(cell), to_1d(new_cell)] = a_prob
            R[to_1d(cell), a_index] = white_cell_reward

    for a_index, action in enumerate(actions):
        for cell in np.ndindex(grid_size):
            # up
            new_cell = (cell[0] - 1, cell[1])
            update_P_and_R(cell, new_cell, a_index, action['up'])

            # down
            new_cell = (cell[0] + 1, cell[1])
            update_P_and_R(cell, new_cell, a_index, action['down'])

            # left
            new_cell = (cell[0], cell[1] - 1)
            update_P_and_R(cell, new_cell, a_index, action['left'])

            # right
            new_cell = (cell[0], cell[1] + 1)
            update_P_and_R(cell, new_cell, a_index, action['right'])

    return P, R, (grid_size, black_cells, green_cell_loc, red_cell_loc)


def visualize_grid_policy(policy, actions, grid):

    arrows = []
    shape = grid[0]
    black = grid[1]
    green = grid[2]
    red = grid[3]

    for p in policy:
        arrows.append(actions[p])

    arrows_array = np.asarray(arrows)
    policy_viz = np.reshape(arrows_array, shape)
    for b in black:
        policy_viz[b] = 'X'
    policy_viz[green] = '*'
    policy_viz[red] = 'R'
    print()
    print(policy_viz)
    policy_df = pd.DataFrame(policy_viz)
    policy_df.to_csv('policy.csv')


def visualize_value_function(values, grid):

    # min_value = np.min(values) - 1
    values_array = np.asarray(values)
    # sb.heatmap(np.reshape(values_array, grid[0]), cmap='coolwarm', vmin=min_value, annot=False, fmt='.0%',
    #            annot_kws={"size": 20})
    sb.heatmap(np.reshape(values_array, grid[0]), cmap='coolwarm', annot=False, fmt='.0%',
               annot_kws={"size": 20})
    plt.show()


def solve_policy_iteration(P, R, grid, details):

    print('\n ----- Policy Iteration -----\n')
    pi = mdptoolbox.mdp.PolicyIteration(P, R, 0.5, max_iter=100, policy0=None, eval_type=0)
    pi.setVerbose()
    pi.run()
    print('\nValue function:', pi.V)
    print('\nPolicy:', pi.policy)
    if grid:
        actions = {0: '^', 1: 'V', 2: '<', 3: '>'}
        visualize_grid_policy(pi.policy, actions, details)
        # visualize_value_function(pi.V, details)
        for i in (0, 10, 20, 30, 50):
            visualize_value_function(pi.utilities[i], details)
        # Visualize policies over time
        # for p in pi.policies:
        #     visualize_grid_policy(p, actions, details)

    print('\nIterations:', pi.iter)
    print('\nTime:', pi.time)
    # print('\nUtilities by iteration:', pi.utilities)
    print()

    # Analyze policies over time
    df_policies = pd.DataFrame(pi.policies)
    df_policies.to_csv('policies.csv')
    # print(df_policies)

    # Analyze utilities over time
    # Analyze utilities over time
    df_utilities = pd.DataFrame(pi.utilities)
    print()
    # print(df_utilities)
    # df_utilities.plot()
    # plt.title('Policy Evaluation for Policy Iteration', fontsize='20')
    # plt.xlabel('Iterations', fontsize='18')
    # plt.ylabel('Utility Values', fontsize='18')
    # plt.xticks(fontsize='16')
    # plt.legend(fontsize='10', loc='lower right')
    # plt.show()
    df_utilities.to_csv('utilities.csv')

    # Analyze n_differences over time
    df_n_differences = pd.DataFrame(pi.n_differences)
    df_n_differences.plot()
    plt.title('Policy Iteration Convergence', fontsize='20')
    plt.xlabel('Iterations', fontsize='18')
    plt.ylabel('n_differences', fontsize='18')
    plt.xticks(fontsize='16')
    plt.legend().remove()
    plt.show()


def solve_value_iteration(P, R, grid, details):

    print('\n ----- Value Iteration -----\n')
    vi = mdptoolbox.mdp.ValueIteration(P, R, 0.5, epsilon=0.0001, max_iter=1000, initial_value=0)
    vi.setVerbose()
    vi.run()
    # print('\nUtilities:', vi.V)
    # print('\nPolicy:', vi.policy)
    if grid:
        actions = {0: '^', 1: 'V', 2: '<', 3: '>'}
        visualize_grid_policy(vi.policy, actions, details)
        # visualize_value_function(vi.V, details)
        for i in (1, 5, 10, -1):
            visualize_value_function(vi.utilities[i], details)
    print('\nIterations:', vi.iter)
    print('\nTime:', vi.time)
    # print('\nUtilities by iteration:', vi.utilities)
    print()

    # Analyze utilities over time
    df_utilities = pd.DataFrame(vi.utilities)
    # print()
    # # print(df_utilities)
    # df_utilities.plot()
    # plt.title('Value Iteration Convergence', fontsize='20')
    # plt.xlabel('Iterations', fontsize='18')
    # plt.ylabel('Utility Values', fontsize='18')
    # plt.xticks(fontsize='16')
    # plt.legend(fontsize='10', loc='lower right')
    # plt.show()
    df_utilities.to_csv('utilities.csv')

    # Analyze policies over time
    df_policies = pd.DataFrame(vi.policies)
    df_policies.to_csv('policies.csv')
    # print(df_policies)

    # Analyze variation over time
    df_variations = pd.DataFrame(vi.variations)
    df_variations.plot()
    plt.title('Value Iteration Convergence', fontsize='20')
    plt.xlabel('Iterations', fontsize='18')
    plt.ylabel('Variance', fontsize='18')
    plt.xticks(fontsize='16')
    plt.legend().remove()
    plt.show()


def solve_qlearner(P, R, grid, details):

    print('\n ----- Q Learner -----\n')
    runs = 1
    policies = []
    for r in range(runs):
        ql = mdptoolbox.mdp.QLearning(P, R, discount=0.9, n_iter=1000000)
        ql.setVerbose()
        ql.run()
        print('\nLearned value function:', ql.V)
        print('\nPolicy:', ql.policy)
        if grid:
            actions = {0: '^', 1: 'V', 2: '<', 3: '>'}
            visualize_grid_policy(ql.policy, actions, details)
            visualize_value_function(ql.V, details)
        print('\nTime:', ql.time)

        # Analyze discrepancies
        df_mean_disc = pd.DataFrame(ql.mean_discrepancy)
        df_mean_disc.plot()
        plt.title('Q-Learner Q Value Variation', fontsize='20')
        plt.xlabel('100x Iterations', fontsize='18')
        plt.ylabel('Mean Discrepancy Values', fontsize='18')
        plt.xticks(fontsize='16')
        # plt.legend(fontsize='10', loc='upper right')
        plt.show()

        # Keep track of policies for multiple independent runs
        policies.append(ql.policy)

    df_policies = pd.DataFrame(policies)
    df_policies.to_csv('policies.csv')


if __name__ == "__main__":

    grid_problem = True
    grid_details = ()

    # Generate aima basic grid MDP
    # P, R = generate_aima_grid()

    # Generate forest management MDP
    # P, R = mdptoolbox.example.forest(S=15, r1=4, r2=2, p=0.1, is_sparse=False)
    # P, R = mdptoolbox.example.forest(S=3, r1=4, r2=2, p=0.1, is_sparse=False)

    # Generate grid world MDP
    P, R, grid_details = generate_grid_world()

    # Solve MDP: Value Iteration
    # solve_value_iteration(P, R, grid_problem, grid_details)

    # Solve MDP: Policy Iteration
    # solve_policy_iteration(P, R, grid_problem, grid_details)

    # Solve MDP: Qlearner
    solve_qlearner(P, R, grid_problem, grid_details)

