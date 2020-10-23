import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rand

'''
Actions:
Left: 0
Right: 1
Up: 2
Down: 3
'''


class Environment():
    def __init__(self, size=(5, 10), default_target=(3, 9)):
        self.default_target = default_target
        self.size = size
        self.grid_position = self.map_grid_to_int()
        self.reward = -1 * np.ones(self.size)
        self.reward[default_target[0], default_target[1]] = 30

    def map_grid_to_int(self):
        '''
        Map each grid location to an integer for easy access
        :return: dictionary
        '''
        output_dict = {}
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                ind = i * self.size[1] + j
                output_dict[ind] = (i, j)
        return output_dict

    def adjacent_cells(self, position=0):
        '''
        Return the adjacent cells from any position in the grid
        :return: list of adjacent cells
        '''
        adjacent_cells = []
        if position not in self.grid_position.keys():
            raise ValueError('The position is out of the grid')

        position_coord = self.grid_position[position]
        state_up = (position_coord[0] - 1) * self.size[1] + position_coord[1]
        state_down = (position_coord[0] + 1) * self.size[1] + position_coord[1]
        state_left = position_coord[0] * self.size[1] + (position_coord[1] - 1)
        state_right = position_coord[0] * self.size[1] + (position_coord[1] + 1)
        for pos in [state_up, state_down, state_left, state_right]:
            In_the_grid1 = pos >= 0 # check that the most left column doesn't have a left adjacent state
            In_the_grid2 = pos in self.grid_position.keys() # check the next state is not out of the grid
            # check that the most right column doesn't have a right adjacent state
            End_not_adjacent_to_next_line1 = position_coord[1] == self.size[1] - 1 and pos == state_right
            End_not_adjacent_to_next_line2 = position_coord[1] == 0 and pos == state_left
            if In_the_grid1 and In_the_grid2:
                if End_not_adjacent_to_next_line1 or End_not_adjacent_to_next_line2:
                    continue
                adjacent_cells.append(pos)
        return adjacent_cells

    def get_reward_and_next_state(self, current_state=0, action=0):
        '''
        Given the current state and the action taken, return the state in which the agent land
        :param current_state:
        :param action:
        :return:
        '''
        if current_state not in self.grid_position.keys():
            raise ValueError('The position is out of the grid')
        position_curr_state = self.grid_position[current_state]
        state_up = (position_curr_state[0] - 1) * self.size[1] + position_curr_state[1]
        state_down = (position_curr_state[0] + 1) * self.size[1] + position_curr_state[1]
        state_left = position_curr_state[0] * self.size[1] + (position_curr_state[1] - 1)
        state_right = position_curr_state[0] * self.size[1] + (position_curr_state[1] + 1)
        adjacent_states = self.adjacent_cells(current_state)
        if action == 0 and state_left in adjacent_states:
            return self.get_reward(state_left), state_left
        elif action == 1 and state_right in adjacent_states:
            return self.get_reward(state_right), state_right
        elif action == 2 and state_up in adjacent_states:
            return self.get_reward(state_up), state_up
        elif action == 3 and state_down in adjacent_states:
            return self.get_reward(state_down), state_down
        return 0, current_state

    def get_reward(self, state=0):
        grid_pos = self.grid_position[state]
        return self.reward[grid_pos[0], grid_pos[1]]

    def update_reward(self, target_state=0):
        '''
        Update the reward signal every time the target moves
        :param target_state:
        :return:
        '''
        self.reward = -1 * np.ones(self.size)
        target_state_coord = self.grid_position[target_state]
        self.reward[target_state_coord[0], target_state_coord[1]] = 30

    def get_default_target(self):
        return list(self.grid_position.values()).index(self.default_target)


class Agent():
    def __init__(self, env, actions=[0, 1, 2, 3], initial_state=0):
        self.initial_state = initial_state
        self.state = self.initial_state
        self.actions = actions
        q_table_shape = (len(env.grid_position.keys()), len(actions))
        self.q_table = np.zeros(q_table_shape)

    def get_action(self, epsilon):
        prob_greedy = rand.uniform(0, 1)
        if prob_greedy < epsilon:
            action = rand.sample(self.actions, 1)[0]
        else:
            action = np.argmax(self.q_table[self.state])
        return action

    def update_qtable(self, action, reward, next_state, alpha, beta):
        self.q_table[self.state, action] = self.q_table[self.state, action] + alpha * (
                reward + beta * np.max(self.q_table[next_state]) - self.q_table[self.state, action]
        )

    def update_state(self, next_state):
        self.state = next_state

    def move_target(self):
        '''
        Moves the agent randomly on the enivronment. Only called when the agent is the target.
        Moving the target also updates the reward table
        :return:
        '''
        action = rand.sample(self.actions, 1)[0]
        _, next_state = env.get_reward_and_next_state(self.state, action)
        env.update_reward(next_state)  # Updating the environment reward
        self.update_state(next_state)


class QLearning():
    def __init__(self, env, epsilon=0.3, beta=0.99, alpha=0.1, episodes=1000):
        self.env = env
        self.epsilon = epsilon
        self.beta = beta
        self.alpha = alpha
        self.episodes = episodes
        # q_table_shape = (len(env.grid_position.keys()), len(actions))
        # self.q_table = []

    def evaluate_agents(self, agent1, agent2, target_agent, sum_reward=False):
        if agent2 == None:
            # Only one agent

            steps = 0
            agent1.state = agent1.initial_state
            target_agent.state = target_agent.initial_state
            trajectory_agent1 = []
            trajectory_target = []
            while (agent1.state != target_agent.state) and (steps < 1000):
                trajectory_agent1.append(list(env.grid_position[agent1.state]))
                agent_action = agent1.get_action(self.epsilon)
                (reward, next_state) = env.get_reward_and_next_state(agent1.state, agent_action)

                agent1.update_state(next_state)

                trajectory_target.append(list(env.grid_position[target_agent.state]))
                target_agent.move_target()
                steps += steps

            trajectory_agent1 = np.array(trajectory_agent1)
            trajectory_target = np.array(trajectory_target)
            policy_agent1 = np.argmax(agent1.q_table, axis=1).reshape(self.env.size)
            value_table_agent1 = np.max(agent1.q_table, axis=1).reshape(self.env.size)

            fig, ax = plt.subplots()
            last_ind = trajectory_target.shape[0] - 1

            # ax.matshow(value_table, cmap=plt.cm.Blues)
            ax.matshow(value_table_agent1, cmap='viridis')

            plt.plot(trajectory_agent1[:, 1], trajectory_agent1[:, 0], color='b')
            plt.scatter(trajectory_agent1[0, 1], trajectory_agent1[0, 0], color='b')
            plt.plot(trajectory_target[:, 1], trajectory_target[:, 0], color='r')
            plt.scatter(trajectory_target[0, 1], trajectory_target[0, 0], color='r')
            plt.scatter(trajectory_target[last_ind, 1], trajectory_target[last_ind, 0], color='black')
            ####
            def action_to_direction(action):
                if action == 0:
                    return "<"
                elif action == 1:
                    return ">"
                elif action == 2:
                    return "^"
                elif action == 3:
                    return "v"

            for i in range(policy_agent1.shape[0]):
                for j in range(policy_agent1.shape[1]):
                    action = policy_agent1[i, j]
                    #if (i, j) == env.grid_position[target_agent.state]:
                    #    c = "T"
                    #elif (i, j) == env.grid_position[target_agent.initial_state]:
                    if (i, j) == env.grid_position[target_agent.initial_state]:
                        c = "S" #+ agent_nber
                    elif env.grid_position[agent1.state] != env.grid_position[target_agent.initial_state] and \
                            (i, j) == env.grid_position[target_agent.initial_state]:
                        c = "A" #+ agent_nber
                    else:
                        c = action_to_direction(action)
                    # ax.text(i, j, c, va='center', ha='center')
                    ax.text(j, i, c, va='center', ha='center')
            ####
            plt.show()

        else:
            # Two agents

            steps = 0
            agent1.state = agent1.initial_state
            agent2.state = agent2.initial_state
            target_agent.state = target_agent.initial_state
            trajectory_agent1 = []
            trajectory_agent2 = []
            trajectory_target = []
            while (agent1.state != target_agent.state) and (agent2.state != target_agent.state) and (steps < 5000):
                # Agent1
                trajectory_agent1.append(list(env.grid_position[agent1.state]))
                agent1_action = agent1.get_action(self.epsilon)
                (ag1_reward, ag1_next_state) = env.get_reward_and_next_state(agent1.state, agent1_action)

                # Agent2
                trajectory_agent2.append(list(env.grid_position[agent2.state]))
                agent2_action = agent2.get_action(self.epsilon)
                (ag2_reward, ag2_next_state) = env.get_reward_and_next_state(agent2.state, agent2_action)

                # Update of both agents
                if sum_reward:
                    ag1_reward = ag1_reward + ag2_reward
                    ag2_reward = ag1_reward + ag2_reward
                agent1.update_qtable(agent1_action, ag1_reward, ag1_next_state, self.alpha, self.beta)
                agent1.update_state(ag1_next_state)
                agent2.update_qtable(agent2_action, ag2_reward, ag2_next_state, self.alpha, self.beta)
                agent2.update_state(ag2_next_state)

                trajectory_target.append(list(env.grid_position[target_agent.state]))
                target_agent.move_target()
                steps += steps

            trajectory_agent1 = np.array(trajectory_agent1)
            trajectory_agent2 = np.array(trajectory_agent2)
            trajectory_target = np.array(trajectory_target)
            policy_agent1 = np.argmax(agent1.q_table, axis=1).reshape(self.env.size)
            value_table_agent1 = np.max(agent1.q_table, axis=1).reshape(self.env.size)
            value_table_agent2 = np.max(agent2.q_table, axis=1).reshape(self.env.size)


            fig, ax = plt.subplots()

            last_ind = trajectory_target.shape[0]-1

            # ax.matshow(value_table, cmap=plt.cm.Blues)
            ax.matshow(value_table_agent1, cmap='viridis')
            plt.plot(trajectory_agent1[:, 1], trajectory_agent1[:, 0], color='b')
            plt.scatter(trajectory_agent1[0, 1],trajectory_agent1[0, 1], color='b')
            plt.plot(trajectory_agent2[:, 1], trajectory_agent2[:, 0], color='g')
            plt.scatter(trajectory_agent2[0, 1], trajectory_agent2[0, 1], color='g')
            plt.plot(trajectory_target[:, 1], trajectory_target[:, 0], color='r')
            plt.scatter(trajectory_target[0, 1], trajectory_target[0, 0], color='r')
            plt.scatter(trajectory_target[last_ind, 1], trajectory_target[last_ind, 0], color='black')
            plt.show()


    def learn_with_one_agent(self, agent, target_agent):
        episodes_reward = []
        for episode in range(self.episodes):
            steps = 0
            agent.state = agent.initial_state
            target_agent.state = target_agent.initial_state
            while (agent.state != target_agent.state) and (steps < 1000):
                agent_action = agent.get_action(self.epsilon)
                (reward, next_state) = env.get_reward_and_next_state(agent.state, agent_action)
                agent.update_qtable(agent_action, reward, next_state, self.alpha, self.beta)
                agent.update_state(next_state)

                target_agent.move_target()
                steps += steps

            episodes_reward.append(reward)

            if episode % 400 == 0:
                try:
                    self.evaluate_agents(agent1, None, target_agent, sum_reward=False)
                    # self.print_value_fct(agent.q_table)
                    '''
                    agent_ini_pos = env.grid_position[agent.initial_state]
                    agent_curr_pos = env.grid_position[agent.initial_state]
                    target_pos = env.grid_position[target_agent.state]
                    #print_value_fct(agent.q_table, agent_pos, target_pos, self.env.size)
                    print_value_fct(agent1.q_table, agent_ini_pos, agent_curr_pos, target_pos, self.env.size)
                    plt.show()
                    '''
                except Exception as e:
                    print(e)
                    continue

        plt.plot(episodes_reward, label='agent', color='b')
        plt.title('Training reward per episode')
        plt.legend()
        plt.show()

    def learn_with_two_agents1(self, agent1, agent2, target_agent):
        episodes_reward_agent1 = []
        episodes_reward_agent2 = []
        for episode in range(self.episodes):
            steps = 0
            agent1.state = agent1.initial_state
            agent2.state = agent2.initial_state
            target_agent.state = target_agent.initial_state
            while (agent1.state != target_agent.state) and (agent2.state != target_agent.state) and (steps < 5000):
                # Agent1
                agent1_action = agent1.get_action(self.epsilon)
                (ag1_reward, ag1_next_state) = env.get_reward_and_next_state(agent1.state, agent1_action)
                agent1.update_qtable(agent1_action, ag1_reward, ag1_next_state, self.alpha, self.beta)
                agent1.update_state(ag1_next_state)

                # Agent2
                agent2_action = agent2.get_action(self.epsilon)
                (ag2_reward, ag2_next_state) = env.get_reward_and_next_state(agent2.state, agent2_action)
                agent2.update_qtable(agent2_action, ag2_reward, ag2_next_state, self.alpha, self.beta)
                agent2.update_state(ag2_next_state)

                target_agent.move_target()
                steps += steps

            episodes_reward_agent1.append(ag1_reward)
            episodes_reward_agent2.append(ag2_reward)

            if episode % 400 == 0:
                try:
                    #self.evaluate_agents(agent1, agent2, target_agent, sum_reward=False)
                    print_value_fct2(env, agent1, agent2, target_agent, output_size=(5, 10), sum_reward=False, epsilon=0.3, alpha=0.1, beta=0.99)
                    '''
                    agent1_ini_pos = env.grid_position[agent1.initial_state]
                    agent2_ini_pos = env.grid_position[agent2.initial_state]
                    agent1_curr_pos = env.grid_position[agent1.state]
                    agent2_curr_pos = env.grid_position[agent2.state]
                    target_pos = env.grid_position[target_agent.state]
                    print_value_fct(agent1.q_table, agent1_ini_pos, agent1_curr_pos, target_pos, self.env.size, 1)
                    print_value_fct(agent2.q_table, agent2_ini_pos, agent2_curr_pos, target_pos, self.env.size, 2)
                    plt.show()
                    '''
                except Exception as e:
                    print(e)
                    continue

        plt.plot(episodes_reward_agent1, label='agent 1', color='b')
        plt.plot(episodes_reward_agent2, label='agent 2', color='g')
        plt.title('Training reward per episode')
        plt.legend()
        plt.show()

    def learn_with_two_agents2(self, agent1, agent2, target_agent):
        # The two agents receive the same reward that is average of what each observed when exploring
        episodes_reward_agent1 = []
        episodes_reward_agent2 = []
        for episode in range(self.episodes):
            steps = 0
            agent1.state = agent1.initial_state
            agent2.state = agent2.initial_state
            target_agent.state = target_agent.initial_state
            while (agent1.state != target_agent.state) and (agent2.state != target_agent.state) and (steps < 5000):
                # Agent1
                agent1_action = agent1.get_action(self.epsilon)
                (ag1_reward, ag1_next_state) = env.get_reward_and_next_state(agent1.state, agent1_action)

                # Agent2
                agent2_action = agent2.get_action(self.epsilon)
                (ag2_reward, ag2_next_state) = env.get_reward_and_next_state(agent2.state, agent2_action)

                # Update of both agents
                reward = ag1_reward + ag2_reward
                agent1.update_qtable(agent1_action, reward, ag1_next_state, self.alpha, self.beta)
                agent1.update_state(ag1_next_state)
                agent2.update_qtable(agent2_action, reward, ag2_next_state, self.alpha, self.beta)
                agent2.update_state(ag2_next_state)

                target_agent.move_target()
                steps += steps

            episodes_reward_agent1.append(ag1_reward)
            episodes_reward_agent2.append(ag2_reward)

            if episode % 400 == 0:
                try:
                    #self.evaluate_agents(agent1, agent2, target_agent, sum_reward=True)
                    print_value_fct2(env, agent1, agent2, target_agent, output_size=(5, 10), sum_reward=True, epsilon=0.3, alpha=0.1, beta=0.99)
                    '''
                    agent1_ini_pos = env.grid_position[agent1.initial_state]
                    agent2_ini_pos = env.grid_position[agent2.initial_state]
                    agent1_curr_pos = env.grid_position[agent1.state]
                    agent2_curr_pos = env.grid_position[agent2.state]
                    target_pos = env.grid_position[target_agent.state]
                    print_value_fct(agent1.q_table, agent1_ini_pos, agent1_curr_pos, target_pos, self.env.size, 1)
                    print_value_fct(agent2.q_table, agent2_ini_pos, agent2_curr_pos, target_pos, self.env.size, 2)
                    plt.show()
                    '''
                except Exception as e:
                    print(e)
                    continue

        plt.plot(episodes_reward_agent1, label='agent 1', color='b')
        plt.plot(episodes_reward_agent2, label='agent 2', color='g')
        plt.title('Training reward per episode [cummulated reward]')
        plt.legend()
        plt.show()


def print_value_fct(qtable, agent_ini_pos, agent_curr_pos, target_pos, output_size=(5, 10), agent_nbr=0):
    policy = np.argmax(qtable, axis=1).reshape(output_size)
    value_table = np.max(qtable, axis=1).reshape(output_size)

    fig, ax = plt.subplots()

    # Customize the value for agent and target position
    value_table[agent_ini_pos] = 100
    value_table[target_pos] = 250
    if agent_curr_pos != target_pos:
        value_table[agent_curr_pos] = 150

    #ax.matshow(value_table, cmap=plt.cm.Blues)
    ax.matshow(value_table, cmap='viridis')

    agent_nber = str(agent_nbr) if agent_nbr != 0 else ''
    def action_to_direction(action):
        if action == 0:
            return "<"
        elif action == 1:
            return ">"
        elif action == 2:
            return "^"
        elif action == 3:
            return "v"

    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            action = policy[i, j]
            if (i,j) == target_pos:
                c = "T"
            elif (i,j) == agent_ini_pos:
                c = "S" + agent_nber
            elif agent_curr_pos != target_pos and (i,j) == agent_curr_pos:
                c = "A" + agent_nber
            else:
                c = action_to_direction(action)
            #ax.text(i, j, c, va='center', ha='center')
            ax.text(j, i, c, va='center', ha='center')
    # print(value_table)
    #plt.show()

def print_value_fct2(env, agent1, agent2, target_agent, output_size=(5, 10), sum_reward=False, epsilon=0.3, alpha=0.1, beta=0.99):
    steps = 0
    agent1.state = agent1.initial_state
    agent2.state = agent2.initial_state
    target_agent.state = target_agent.initial_state
    trajectory_agent1 = []
    trajectory_agent2 = []
    trajectory_target = []
    while (agent1.state != target_agent.state) and (agent2.state != target_agent.state) and (steps < 5000):
        # Agent1
        trajectory_agent1.append(list(env.grid_position[agent1.state]))
        agent1_action = agent1.get_action(epsilon)
        (ag1_reward, ag1_next_state) = env.get_reward_and_next_state(agent1.state, agent1_action)

        # Agent2
        trajectory_agent2.append(list(env.grid_position[agent2.state]))
        agent2_action = agent2.get_action(epsilon)
        (ag2_reward, ag2_next_state) = env.get_reward_and_next_state(agent2.state, agent2_action)

        # Update of both agents
        if sum_reward:
            ag1_reward = ag1_reward + ag2_reward
            ag2_reward = ag1_reward + ag2_reward
        agent1.update_qtable(agent1_action, ag1_reward, ag1_next_state, alpha, beta)
        agent1.update_state(ag1_next_state)
        agent2.update_qtable(agent2_action, ag2_reward, ag2_next_state, alpha, beta)
        agent2.update_state(ag2_next_state)

        trajectory_target.append(list(env.grid_position[target_agent.state]))
        target_agent.move_target()
        steps += steps

    fig, axs = plt.subplots(2,1)
    for agent, ax, trajectory, ag_nbr, col in zip(
            [agent1, agent2], axs, [trajectory_agent1, trajectory_agent2], [1, 2], ['b', 'g']
    ):
        policy = np.argmax(agent.q_table, axis=1).reshape(output_size)
        value_table = np.max(agent.q_table, axis=1).reshape(output_size)

        value_table[env.grid_position[agent.initial_state]] = 100
        value_table[env.grid_position[target_agent.initial_state]] = 250
        if env.grid_position[agent.state] != env.grid_position[target_agent.state]:
            value_table[env.grid_position[agent.state]] = 150
        '''
        #ax.matshow(value_table, cmap=plt.cm.Blues)
        ax.matshow(value_table, cmap='viridis')

        #agent_nber = str(agent_nbr) if agent_nbr != 0 else ''
        def action_to_direction(action):
            if action == 0:
                return "<"
            elif action == 1:
                return ">"
            elif action == 2:
                return "^"
            elif action == 3:
                return "v"

        for i in range(policy.shape[0]):
            for j in range(policy.shape[1]):
                action = policy[i, j]
                if (i,j) == env.grid_position[target_agent.state]:
                    c = "T"
                elif (i,j) == env.grid_position[agent.initial_state]:
                    c = "S" + str(ag_nbr)
                elif env.grid_position[agent.state] != env.grid_position[target_agent.state] and \
                        (i,j) == env.grid_position[agent.state]:
                    c = "A" + str(ag_nbr)
                else:
                    c = action_to_direction(action)
                #ax.text(i, j, c, va='center', ha='center')
                ax.text(j, i, c, va='center', ha='center')
        '''
        trajectory = np.array(trajectory)
        trajectory_target = np.array(trajectory_target)

        last_ind = trajectory_target.shape[0] - 1

        # ax.matshow(value_table, cmap=plt.cm.Blues)
        ax.matshow(value_table, cmap='viridis')
        ax.set_title('Trajectory of Target and Agent '+str(ag_nbr))

        ax.plot(trajectory[:, 1], trajectory[:, 0], color=col)
        ax.scatter(trajectory[0, 1], trajectory[0, 0], color=col)
        ax.plot(trajectory_target[:, 1], trajectory_target[:, 0], color='r')
        ax.scatter(trajectory_target[0, 1], trajectory_target[0, 0], color='r')
        ax.scatter(trajectory_target[last_ind, 1], trajectory_target[last_ind, 0], color='black')

        #plt.show()
    # print(value_table)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    env_size = (5, 10)
    env = Environment(env_size)
    #initial_agent_state = 0
    initial_agent_state = rand.sample(list(env.grid_position.keys()), 1)[0]
    experiment = 1

    agent1 = Agent(env, actions=[0, 1, 2, 3], initial_state=initial_agent_state)
    agent2 = Agent(env, actions=[0, 1, 2, 3], initial_state=initial_agent_state)
    start_state_target = env.get_default_target()
    target_agent = Agent(env, actions=[0, 1, 2, 3], initial_state=start_state_target)

    qlearn = QLearning(env)
    if experiment == 0:
        # One agent
        qlearn.learn_with_one_agent(agent1, target_agent)
    elif experiment == 1:
        # Two agents not sharing rewards
        qlearn.learn_with_two_agents1(agent1, agent2, target_agent)
    else:
        # Two agents sharing rewards
        qlearn.learn_with_two_agents2(agent1, agent2, target_agent)

    #print(np.fromiter(env.grid_position.keys(), dtype=float).reshape(env_size))
    #print(env.adjacent_cells(49))
    # env.update_reward(25)
    # print(env.reward)
