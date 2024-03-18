indices = [];
range = 0:0.1:1;
n = length(range);
rewards = zeros(n, n, n); % Initialize rewards matrix
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

inputSize = 3; % Assuming the state is represented by 3 values
outputSize = 10; %Set of actions
net = createDeepQNetwork(inputSize, outputSize);
options = trainingOptions('adam', ...
        'MaxEpochs',1, ...
        'MiniBatchSize', 32, ...
        'Verbose', false);

episodes = 100;
gamma = 0.9; % Discount factor
epsilon = 0.5; % Exploration rate
alpha = 0.001; % Learning rate = or 0.01;
%miniBatchSize = 10;

for episode = 1:episodes
    disp("current episode: " + episode);
    count = 1;   % will be a usefull flag for iteration/episode, initialize to 0/False at start of new episode
    % Get starting state
    [W_d_index, W_l_index, W_ec_index] = Get_Starting_State(index_rewards_table); 

    while (count <= 100)
        disp("current count: " + count);
        state = [W_d_index, W_l_index, W_ec_index];

        predicted_action = Get_Predicted_Action(net,state, epsilon);

        % Take a step in the environment using the above predicted action.
        [new_W_d_Index, new_W_l_Index, new_W_ec_Index] = Get_Next_State(W_d_index, W_l_index, W_ec_index, predicted_action);  
        
        % Calculate reward for jumping into new state using predicted action.
        new_index_reward = rewards(new_W_d_Index, new_W_l_Index, new_W_ec_Index);
        
        % Calculate target Q-value: For each experience tuple, compute the target Q-value as reward + gamma * max(Q_values(next_state, :)),
        % where Q_values(next_state, :) are the Q-values predicted by the network for the next state.
        
        next_State = [new_W_d_Index new_W_l_Index new_W_ec_Index];
        next_State = next_State';                                         % Transpose the state to make ready for DQN input.
        new_state_DL = dlarray(next_State, 'CB');                         % Converting the input(=nextState) into 
        next_Q_predict = predict(net, new_state_DL);                      % Exploit: Use the network to predict next Q-values
        % also need to assign "- infinity" to rejected actions for the next_State as well
        target_Q = new_index_reward + gamma * max(next_Q_predict, [], 1); % gamma is the discount factor, which represents the importance of future rewards relative to immediate rewards.
        
        % Update the model
        X = dlarray(state', 'CB');       % Current Input(state)
        Y = dlarray(target_Q, 'CB');    
        [loss, gradients] = dlfeval(@modelGradients, net, X, Y, predicted_action);
        % Print the loss to monitor training progress
        %disp("Loss: " + loss);
        net = dlupdate(@(w, dw) w + alpha * dw, net, gradients);      
        
        % Update state (setting: current state <- new state) OR % Move to the next state
        W_d_index = new_W_d_Index;
        W_l_index = new_W_l_Index;
        W_ec_index = new_W_ec_Index;

        count = count +1;
    end
end

function net = createDeepQNetwork(inputSize, outputSize)
    layers = [
        featureInputLayer(inputSize, 'Normalization', 'none', 'Name', 'input')
        fullyConnectedLayer(64, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(64, 'Name', 'fc2')
        reluLayer('Name', 'relu2')
        fullyConnectedLayer(64, 'Name', 'fc3')
        reluLayer('Name', 'relu3')
        fullyConnectedLayer(64, 'Name', 'fc4')
        reluLayer('Name', 'relu4')
        fullyConnectedLayer(outputSize, 'Name', 'output')];
    lgraph = layerGraph(layers);
    net = dlnetwork(lgraph);
end

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

function Predicted_Action = Get_Predicted_Action(net,state, epsilon)
    validActions = [];

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

        % After calculating new_W_d_Index, new_W_l_Index, new_W_ec_Index
        % Check if any index is less than 1 or greater than 11
        if new_W_d_Index < 1 || new_W_l_Index < 1 || new_W_ec_Index < 1 || new_W_d_Index > 11 || new_W_l_Index > 11 || new_W_ec_Index > 11
            %disp("dont update validActions array...");
        else
            %disp("updating validActions array...");
            validActions = [validActions, actionTry];     % True = 1
        end
    end
    disp("current state: ");
    disp(state);
    disp("its Valid Actions: ");
    disp(validActions)

        % Exploration: choose a random valid action
        if rand() < epsilon
            disp("Exploring as, in this count, rand() < epsilon ");
            random_action_Index = randi(length(validActions));
            Predicted_Action = validActions(random_action_Index);
        else
            disp("Exploiting as, in this count, rand() > epsilon ");
            rejected_Actions = setdiff(1:10, validActions);         % computing rejected_Actions, Assuming 10 possible actions
            state = state';                                         % Transpose the state to make ready for DQN input.
            state_DL = dlarray(state, 'CB');                        % Converting the input(=state) into 
            Q_predict = predict(net, state_DL);                     % Exploit: Use the network to predict Q-values
            Q_predict(rejected_Actions)=-inf;                       % Initializing rejected_Actions as -Infinity.
            disp("Printing predicted Q values:Q_predict...");
            disp(Q_predict);
            % Assign value 1 to cells that have NaN
            Q_predict(isnan(Q_predict)) = -100;
            [pred_current_max_reward, Predicted_Action] = max(Q_predict, [], 1);   % Choose the action with the highest Q-value(=pred_current_max_reward)
            %disp("current predicted maximum reward: " + pred_current_max_reward);
        end
end

function [new_W_d_Index, new_W_l_Index, new_W_ec_Index] = Get_Next_State(W_d_index, W_l_index, W_ec_index, action)
    step = 0.1;
    % Action to state transition logic
    disp("choosen action from the above valid actions: ")
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

function [loss, gradients] = modelGradients(net, X, Y, predicted_action)
    Y_pred = forward(net, X);
    % Select the Q-value for the taken action
    Q = Y_pred(predicted_action);
    loss = mse(Y, Q);
    gradients = dlgradient(loss, net.Learnables);
end
