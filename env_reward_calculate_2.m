% Environment Setup: 
% First, lets define the environment, including the state space and the reward matrix.
indices = [];
range = 0:0.1:1;
n = length(range);
rewards = zeros(n, n, n); % Initialize rewards matrix
%Instead reward table initialization, if you want to load from memory than
%loaded_data = load('E:\Research\sem_8\code\matlab\reward_table.mat');
%rewards = loaded_data.rewards;     % now, we got the reward table

% If you want to save rweward table than 
%save('E:\Research\sem_8\code\matlab\reward_table.mat', 'rewards');

% Assuming A_t (average latency) is calculated elsewhere and available
A_t = 20; % Placeholder value
% A_t_imaginary = 10000;

for W_d_index = 1:n
    for W_l_index = 1:n
        for W_ec_index = 1:n
            W_d = range(W_d_index);
            W_l = range(W_l_index);
            W_ec = range(W_ec_index);       
            if W_d + W_l + W_ec == 1
                % Update rewards based on A_t
                rewards(W_d_index, W_l_index, W_ec_index) = 1 / A_t;
                indices = [indices; W_d_index, W_l_index, W_ec_index];
                %else 
                %rewards(W_d_index, W_l_index, W_ec_index) = 1 / A_t_imaginary;
            end
        end
    end
end
% Create a table from the indices array
index_rewards_table = array2table(indices, 'VariableNames', {'W_d_I', 'W_l_I', 'W_ec_I'});

% Initialize Q_values
Q_values = zeros(n, n, n, 10); % Since 10 possible actions defined in Get_Next_State function
epsilon = 0.9; % Exploration rate
gamma = 0.9; % Discount factor
alpha = 0.9; % Learning rate

% Agent Training:Implement the agent training loop, updating Q-values based on actions taken and rewards received.
for episode = 1:1000 % Number of episodes
    disp("current episodes.....");
    disp(episode);

    % Get starting state
    [W_d_index, W_l_index, W_ec_index] = Get_Starting_State(index_rewards_table);   
    %If you want a simplify verion, instead of Get_Starting_State functon, use a random function that
    %fetches a row from index_rewards_table randomly
    
    action = Get_Next_Action(Q_values, [W_d_index, W_l_index, W_ec_index], epsilon);
    [new_W_d_Index, new_W_l_Index, new_W_ec_Index] = Get_Next_State(W_d_index, W_l_index, W_ec_index, action);  
        
    % Calculate reward for jumping into new state
    new_index_reward = rewards(new_W_d_Index, new_W_l_Index, new_W_ec_Index);
        
    % Update Q-values accordingly
    old_Q_value = Q_values(W_d_index, W_l_index, W_ec_index, action);
    TD = new_index_reward + gamma * max(Q_values(W_d_index, W_l_index, W_ec_index, :)) - old_Q_value;
    new_Q_value = old_Q_value + alpha * TD;
    Q_values(W_d_index, W_l_index, W_ec_index, action) = new_Q_value;
        
    % Update state (setting: current state <- new state)
    W_d_index = new_W_d_Index;
    W_l_index = new_W_l_Index;
    W_ec_index = new_W_ec_Index;
end

%This function generates a random starting state. It returns the indices for W_d, W_l, and W_ec within the range matrix.

function [W_d_index, W_l_index, W_ec_index] = Get_Starting_State(index_rewards_table)

        % Randomly select indices for W_d, W_l, and W_ec
        numRows_I_table = size(index_rewards_table, 1);
        randomIndex = randi([1, numRows_I_table]);
        randomRow_I_table = index_rewards_table(randomIndex, :);
        W_d_I = randomRow_I_table(1, 1);
        W_d_index = W_d_I{1,1};
        W_l_I = randomRow_I_table(1, 2);
        W_l_index = W_l_I{1,1};
        W_ec_I = randomRow_I_table(1, 3);
        W_ec_index = W_ec_I{1,1};
end


%Action Selection: Define the function to select an action based on the current state and epsilon for exploration.

function action = Get_Next_Action(Q_values, state, epsilon)
    validActions = [];
    %disp("validActions contains after set to zero...")
    
    for actionTry = 1:10
        if actionTry==1
            new_W_d_Index = state(1) + 1;  % it is as equal as assigning new_W_d = W_d + 0.1
            new_W_l_Index = state(2) + 1;  % it is as equal as assigning new_W_l = W_l + 0.1
            new_W_ec_Index = state(3) -2; % it is as equal as assigning new_W_ec = W_ec - 0.2
        end
        if actionTry==2
            new_W_d_Index = state(1) + 1;
            new_W_l_Index = state(2) - 1;
            new_W_ec_Index = state(3);
        end
        if actionTry==3
            new_W_d_Index = state(1) + 1;
            new_W_l_Index = state(2) - 2;
            new_W_ec_Index = state(3) + 1;
        end
        if actionTry==4
            new_W_d_Index = state(1) + 1;
            new_W_l_Index = state(2);
            new_W_ec_Index = state(3) - 1;
        end

        if actionTry==5
            new_W_d_Index = state(1) - 1;
            new_W_l_Index = state(2) + 1;
            new_W_ec_Index = state(3);
        end
        
        if actionTry==6
            new_W_d_Index = state(1) - 1;
            new_W_l_Index = state(2) - 1;
            new_W_ec_Index = state(3) + 2;
        end
        
        if actionTry==7 % decrement W_d & increment W_ec
            new_W_d_Index = state(1) - 1;
            new_W_l_Index = state(2);
            new_W_ec_Index = state(3) + 1;
        end

        if actionTry==8 % decrement W_d & decrement W_ec
            new_W_d_Index = state(1) - 1;
            new_W_l_Index = state(2) + 2;
            new_W_ec_Index = state(3) - 1;
        end
        if actionTry==9 % Stay in W_d & increment W_l
            new_W_d_Index = state(1);      % it is as equal as assigning new_W_d = W_d
            new_W_l_Index = state(2) + 1;
            new_W_ec_Index = state(3) - 1;
        end
        if actionTry==10 % Stay in W_d & decrement W_l
            new_W_d_Index = state(1);   
            new_W_l_Index = state(2) - 1;
            new_W_ec_Index = state(3) + 1;
        end
        disp("Printing new_W_d_Index, new_W_l_Index, new_W_ec_Index for each action in action function...");
        disp(new_W_d_Index);
        disp(new_W_l_Index);
        disp(new_W_ec_Index);
        % After calculating new_W_d_Index, new_W_l_Index, new_W_ec_Index
        % Check if any index is less than 1 or greater than 11
        if new_W_d_Index < 1 || new_W_l_Index < 1 || new_W_ec_Index < 1 || new_W_d_Index > 11 || new_W_l_Index > 11 || new_W_ec_Index > 11
            disp("dont update validActions array...");
        else
            disp("updating validActions array...");
            validActions = [validActions, actionTry];     % True = 1
            disp(validActions)
        end
    end
    
    if rand() < epsilon
        % Exploration: choose a random valid action
        random_action_Index = randi(length(validActions));
        action = validActions(random_action_Index);
    else
        % Exploitation: choose the best valid action from Q_values
        validQValues = Q_values(state(1), state(2), state(3), validActions);
        [max_reward, maxIndex] = max(validQValues);
        disp('current maximum reward:...')
        disp(max_reward)
        action = validActions(maxIndex);
        %disp(reshape(Q_values(state(1), state(2), state(3),:), [1, 10]));
    end
end

%State Transition:Define the function to get the next state based on the current state and action taken.

function [new_W_d_Index, new_W_l_Index, new_W_ec_Index] = Get_Next_State(W_d_index, W_l_index, W_ec_index, action)
    step = 0.1;
    % Action to state transition logic
    disp("action taken as action...")
    disp(action)
    switch action

        case 1 % increment both W_d & W_l
            new_W_d_Index = W_d_index + 1;  % it is as equal as assigning new_W_d = W_d + 0.1
            new_W_l_Index = W_l_index + 1;  % it is as equal as assigning new_W_l = W_l + 0.1
            new_W_ec_Index = W_ec_index -2; % it is as equal as assigning new_W_ec = W_ec - 0.2
        
        case 2 % increment W_d & decrement W_l
            new_W_d_Index = W_d_index + 1;
            new_W_l_Index = W_l_index - 1;
            new_W_ec_Index = W_ec_index;
        
        case 3 % increment both W_d & W_ec
            new_W_d_Index = W_d_index + 1;
            new_W_l_Index = W_l_index - 2;
            new_W_ec_Index = W_ec_index + 1;

        case 4 % increment W_d & decrement W_ec
            new_W_d_Index = W_d_index + 1;
            new_W_l_Index = W_l_index;
            new_W_ec_Index = W_ec_index - 1;

        case 5 % decrement W_d & increment W_l
            new_W_d_Index = W_d_index - 1;
            new_W_l_Index = W_l_index + 1;
            new_W_ec_Index = W_ec_index;

        case 6 % decrement W_d & decrement W_l
            new_W_d_Index = W_d_index - 1;
            new_W_l_Index = W_l_index - 1;
            new_W_ec_Index = W_ec_index + 2;

        case 7 % decrement W_d & increment W_ec
            new_W_d_Index = W_d_index - 1;
            new_W_l_Index = W_l_index;
            new_W_ec_Index = W_ec_index + 1;

        case 8 % decrement W_d & decrement W_ec
            new_W_d_Index = W_d_index - 1;
            new_W_l_Index = W_l_index + 2;
            new_W_ec_Index = W_ec_index - 1;

        case 9 % Stay in W_d & increment W_l
            new_W_d_Index = W_d_index;      % it is as equal as assigning new_W_d = W_d
            new_W_l_Index = W_l_index + 1;
            new_W_ec_Index = W_ec_index - 1;
        
        case 10 % Stay in W_d & decrement W_l
            new_W_d_Index = W_d_index;   
            new_W_l_Index = W_l_index - 1;
            new_W_ec_Index = W_ec_index + 1;
    end
end
