% Fjona Lutaj, Soranna Bacanu, Sakshi Singh, Cristina López Ruiz 


clear all
clc
close all
%%
load('monkeydata_training.mat');
rng(2013);
ix = randperm(length(trial));
trainingData = trial(ix(1:50),:);
testData = trial(ix(51:end),:);


meanSqError = 0;
n_predictions = 0;  

figure
hold on
axis square
grid

% Train Model
tic
modelParameters = positionEstimatorTraining(trainingData)
trueDirections = [];
predictedDirections = [];

for tr=1:size(testData,1)
    display(['Decoding block ',num2str(tr),' out of ',num2str(size(testData,1))]);
    pause(0.001)
    for direc=randperm(8) 
        decodedHandPos = [];

        times=320:20:size(testData(tr,direc).spikes,2);
        
        for t=times
            past_current_trial.trialId = testData(tr,direc).trialId;
            past_current_trial.spikes = testData(tr,direc).spikes(:,1:t); 
            past_current_trial.decodedHandPos = decodedHandPos;

            past_current_trial.startHandPos = testData(tr,direc).handPos(1:2,1); 
            
            if nargout('positionEstimator') == 4
                [decodedPosX, decodedPosY, newParameters, predicted] = positionEstimator(past_current_trial, modelParameters);
                modelParameters = newParameters;
            elseif nargout('positionEstimator') == 2
                [decodedPosX, decodedPosY] = positionEstimator(past_current_trial, modelParameters);
            end
            
            decodedPos = [decodedPosX; decodedPosY];
            decodedHandPos = [decodedHandPos decodedPos];
            trueDirections = [trueDirections; direc];
            predictedDirections = [predictedDirections; predicted];

            meanSqError = meanSqError + norm(testData(tr,direc).handPos(1:2,t) - decodedPos)^2;
            
        end
        n_predictions = n_predictions+length(times);
        hold on
        plot(decodedHandPos(1,:),decodedHandPos(2,:), 'r');
        plot(testData(tr,direc).handPos(1,times),testData(tr,direc).handPos(2,times),'b')
    end
end
toc
legend('Decoded Position', 'Actual Position')
correctPredictions = sum(trueDirections == predictedDirections);
totalPredictions = length(trueDirections);
accuracy = (correctPredictions / totalPredictions) * 100;
fprintf('Angle classification accuracy: %.2f%%\n', accuracy);

RMSE = sqrt(meanSqError/n_predictions)
%%
function  modelParameters = positionEstimatorTraining(trainingData)

%%%%%% Initialise parameters %%%%%%%
directions = 8;
traininglen =  length(trainingData); 
group = 20;
ema_decay = 0.35; %arbitrary value decided through multiple trials

%%%%%% 1. Preprocess data  %%%%%%
trialFinal = process_and_smooth(trainingData, group, ema_decay);

%%%%%% 2. Rearrange Data for Analysis %%%%%%
modelParameters = struct;
start = 320;
finish = 560;

timepoints = [start:group:finish]/group;
neuron_num = size(trialFinal(1,1).rates,1);

count = 1;
for time = timepoints
    for i = 1: traininglen
        for j = 1: directions
            for k = 1: time
                colIndex = traininglen * (j - 1) + i;
                rowIndexStart = neuron_num * (k - 1) + 1;
                rowIndexEnd = neuron_num * k;
                neuraldata(rowIndexStart:rowIndexEnd,colIndex) = trialFinal(i,j).rates(:,k);     
            end
        end
    end

%%%%%% 3. Most Discriminant Feature Analysis : PCA + LDA %%%%%%

    pca_dimension = 40; % arbitrary value decided throughout multiple trials 
    lda_dimension = 6;  % arbitrary value decided throughout multiple trials 
    [output, weight] = MDF(neuraldata, directions, traininglen, pca_dimension, lda_dimension);


%%%%%% 4. NB Classifier %%%%%%
    modelParameters.classify(count).wTrain = weight;
    modelParameters.classify(count).wTest= output;
    modelParameters.classify(count).mFiring = mean(neuraldata,2);

    % Labels for training data
    labels = repmat(1:directions, traininglen, 1); % Create labels for each direction
    labels = labels(:); % Flatten the labels matrix to create a label vector

    % Train the NB model using PCA and LDA output features
    modelParameters.classify(count).naiveBayesModel = fitcnb(weight', labels);
    % Increment 'count' to move to the next set of parameters or the next fold. This is essential for storing parameters
    % across multiple iterations of training and validation, especially in cross-validation scenarios.

    count = count+1;

end

%%%%%% 5. PCR to get hand positions %%%%%%

% The goal of using Principal Component Regression (PCR) here is to estimate hand positions from neural data. PCR allows us to
% deal effectively with the high dimensionality of neural firing rate data.
% By first applying Principal Component Analysis (PCA) to reduce the dimensionality of the data, we focus on the most significant
% variance components. Then, we use these principal components in a regression model to predict hand positions.


% Create supervised labels for each direction, which will be used in the PCR model. 
labels = repmat(1:directions, traininglen, 1); % Repeat each direction label 'traininglen' times.
labels = labels(:)'; 

% Initialize and resample hand position matrices to ensure consistency in input dimensions.
% This function returns both the original formatted hand positions and their resampled versions for specified time intervals.

[xTestingInterval, yTestingInterval, formattedX, formattedY] = initializeHandPositionMatrices(trainingData, traininglen, directions, group, start, finish);

% Generate a time division array using the Kronecker product to replicate the time interval for each neuron.

timeDivision = kron(group:group:finish, ones(1, neuron_num)); 
Interval = start:group:finish;

% Loop through each direction to model hand positions separately for each.

for directionIndex = 1: directions

    % Extract the current direction's hand position data for all trials.

    currentXPositions = xTestingInterval(:,:,directionIndex);
    currentYPositions = yTestingInterval(:,:,directionIndex);

 % Loop through each time window to calculate regression coefficients.
    % These coefficients are used to predict hand positions from neural data.

    for timeWindowIndex = 1:((finish-start)/group)+1

         % Calculate regression coefficients and the windowed firing rates for the current time window and direction.

        [regressionCoefficientsX, regressionCoefficientsY, windowedFiring] = calcRegressionCoefficients(timeWindowIndex, timeDivision, labels, directionIndex, neuraldata, pca_dimension, Interval, currentXPositions, currentYPositions);

         % Store the calculated regression coefficients and the mean windowed firing rates in the model parameters structure.

        modelParameters.pcr(directionIndex,timeWindowIndex).bx = regressionCoefficientsX;
        modelParameters.pcr(directionIndex,timeWindowIndex).by = regressionCoefficientsY;
        modelParameters.pcr(directionIndex,timeWindowIndex).fMean = mean(windowedFiring,1);

        % Store the average hand positions across all trials for each time window.
        % These averages can be useful for evaluating the model's performance.

        modelParameters.averages(timeWindowIndex).avX = squeeze(mean(formattedX,1));
        modelParameters.averages(timeWindowIndex).avY = squeeze(mean(formattedY,1));
   
    end

end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%  HELPER FUNCTIONS   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function processed_data = process_and_smooth(trial, group, ema_decay)
% Processes trial data by re-binning, sqrt transformation, and smoothing firing rates with 
% an exponential moving average (EMA).

% Inputs:
%  trial = input struct with spikes and handPos data.
%  group = binning resolution (ms).
%  ema_decay = decay factor for EMA, controlling the degree of smoothing.

    % Initialize the output structure
    processed_data = struct;
    
    % Loop through each trial and reaching angle
    for i = 1:size(trial, 2)
        for j = 1:size(trial, 1)

            % Re-bin spikes
            all_spikes = trial(j, i).spikes;
            neuron_num = size(all_spikes, 1);
            points = size(all_spikes, 2);
            t_new = 1:group:points + 1;
            spikes = zeros(neuron_num, numel(t_new) - 1);

            for k = 1:length(t_new) - 1
                if k == length(t_new) - 1
                    spikes(:, k) = sum(all_spikes(:, t_new(k):end), 2); % Ensure we include all remaining data in the last bin
                else
                    spikes(:, k) = sum(all_spikes(:, t_new(k):t_new(k + 1) - 1), 2);
                end
            end

            % Apply sqrt transformation 
            spikes = sqrt(spikes);
            
            % Apply EMA smoothing
            ema_rates = zeros(size(spikes));
            for n = 1: neuron_num
                for t = 2:size(spikes, 2)
                    ema_rates(n, t) = ema_decay * spikes(n, t) + (1 - ema_decay) * ema_rates(n, t - 1);
                end
            end
            
            % Store processed data in the output structure
            processed_data(j, i).rates = ema_rates / (group / 1000); % Adjust rates to spikes per second
            
        end
    end
end

function [output, weight] = MDF(neuraldata, directions, traininglen, pca_dimension, lda_dimension)

    % Most Discriminant Feature Method:
    % This function performs dimensionality reduction on neural data by first applying PCA to reduce the feature space, 
    % followed by LDA to maximize class separability across different directions or classes. 

    % Input:
    %  neuraldata: Matrix containing the neural firing data, with neurons in rows and time in columns
    %  directions: The number of distinct directions or classes in the data
    %  traininglen: The length of the training data for each direction
    %  pca_dimension: The number of principal components to retain in PCA
    %  lda_dimension: The number of dimensions to retain after performing LDA
    
    % Output:
    %  output: The projected data after applying PCA and LDA for dimensionality reduction
    %  weight: The weights calculated for the most discriminant features
   
    
    % Extract principal components from firing data
    U = getPCA(neuraldata);
    princComp = U(:, 1:pca_dimension); % Select the first pca_dimension principal components
    
    % Initialize matrix to store mean vectors for each direction
    mat_between = zeros(size(neuraldata,1), directions);
    
    % Calculate mean vectors for each direction
    for i = 1:directions
        startind = traininglen*(i-1)+1;
        endind = i*traininglen;
        mat_between(:,i) = mean(neuraldata(:, startind:endind), 2);
    end

    % Initialize scatter matrices
    scatter_within = zeros(size(neuraldata, 1));
    scatter_between = zeros(size(neuraldata, 1));
    
    for i = 1:directions
        % Indices for the current direction
        ind = (i-1)*traininglen + 1 : i*traininglen;
        
        % Mean of the current direction
        directionMean = mean(neuraldata(:, ind), 2);
        
        % Scatter within for current direction
        deviation_within = neuraldata(:, ind) - directionMean;
        scatter_within = scatter_within + deviation_within * deviation_within';
        
        % Scatter between calculation
        deviation_between = directionMean - mean(neuraldata, 2);
        scatter_between = scatter_between + traininglen * (deviation_between * deviation_between');
    end
    
    % Perform LDA on the projected data (PCA reduced)
    projectWithin = princComp' * scatter_within * princComp;
    projectBetween = princComp' * scatter_between * princComp;
    
    [eigenvect, eigenval] = eig(pinv(projectWithin) * projectBetween);
    
    % Sort eigenvalues and eigenvectors in descending order
    [~, sortedComp] = sort(diag(eigenval), 'descend');
    
    % Select the lda_dimension dimensions that maximize class separability
    output = princComp * eigenvect(:, sortedComp(1:lda_dimension));
    
    % Calculate the optimum projection using the Most Discriminant Feature method
    weight = output' * (neuraldata - mean(neuraldata, 2));
end


function coeff = getPCA(neuraldata)

% Principal Component Analysis (PCA) is utilized here to reduce the dimensionality of the neural data while retaining as much
% variability in the data as possible. This is crucial in neural data analysis, where datasets often contain a large number
% of variables (neural features) relative to the number of observations (trial counts). 

    % Center the data by subtracting the mean of each feature
    neuralDataCentered = bsxfun(@minus, neuraldata, mean(neuraldata, 2));

    % Compute the covariance matrix from the centered data
    C = neuralDataCentered' * neuralDataCentered;

    % Perform eigenvalue decomposition on the covariance matrix
    % V contains the eigenvectors (principal directions), and D contains the eigenvalues on its diagonal
    [V, D] = eig(C);
    
    % Sort the eigenvalues (and corresponding eigenvectors) in descending order.
    [eigenvalues, order] = sort(diag(D), 'descend');
    V = V(:, order);
    
    % Compute the PCA coefficients (scores) by projecting the original centered data onto the principal component axes.
    coeff = neuralDataCentered * V * diag(1./sqrt(eigenvalues));

end

function [xTestingInterval, yTestingInterval, formattedX, formattedY] = initializeHandPositionMatrices(neuraldata, traininglen, directions, group, start, finish)

% This function initializes and resamples hand position matrices based on neural data.
% It adjusts the size of each trial's hand position data to match the longest trial,
% ensuring consistent input dimensions for subsequent analyses.

% Input:

%   neuraldata : A struct array containing the neural data, where each element
%                has a field 'handPos' corresponding to hand position data.
%   traininglen : The number of training trials.
%   directions : The number of movement directions 
%   group : The binning interval (in milliseconds) used for resampling.
%   start : The start time (in milliseconds) for the analysis interval.
%   finish : The end time (in milliseconds) for the analysis interval.


% Output:

%   xTestingInterval : The resampled x-coordinates of hand positions.
%   yTestingInterval : The resampled y-coordinates of hand positions.
%   formattedX : The x-coordinates of hand positions, adjusted for length and reshaped.
%   formattedY : The y-coordinates of hand positions, adjusted for length and reshaped.

    % Determine the maximum trajectory size across all trials

    maxTrajectorySize = max(arrayfun(@(x) size(x.handPos, 2), neuraldata(:)));

    % Initialize formatted position matrices

    formattedX = zeros(traininglen, maxTrajectorySize, directions);
    formattedY = zeros(traininglen, maxTrajectorySize, directions);

    % Loop over each direction and trial
    for i = 1:directions
        for j = 1:traininglen
            currentSize = size(neuraldata(j, i).handPos, 2);
            paddingSize = maxTrajectorySize - currentSize;

         % Extract the hand positions for the current trial
        currentX = neuraldata(j, i).handPos(1, :);
        currentY = neuraldata(j, i).handPos(2, :);
        
       % If padding is necessary, replicate the last position element
          if paddingSize > 0
            % Replicate the last element for padding
            padX = repmat(currentX(end), 1, paddingSize);
            padY = repmat(currentY(end), 1, paddingSize);
            
            % Combine the original data with the padding
            formattedX(j, :, i) = [currentX, padX];
            formattedY(j, :, i) = [currentY, padY];
          else
            % If no padding is needed, just copy the data
            formattedX(j, :, i) = currentX;
            formattedY(j, :, i) = currentY;
          end
        end
    end

    % Resample the formatted matrices to create intervals based on the 'group' parameter
    % This step aligns the data with the temporal analysis window specified by 'start' and 'finish'
    resampledIndices = start:group:finish;
    xTestingInterval = formattedX(:, resampledIndices, :);
    yTestingInterval = formattedY(:, resampledIndices, :);
end


function [regressionCoefficientsX, regressionCoefficientsY, FilteredFiring ] = calcRegressionCoefficients(timeWindowIndex, timeDivision, labels, directionIndex, neuraldata, pca_dimension, Interval, currentXPositions, currentYPositions)

% This function calculates regression coefficients for predicting hand positions
% from neural data using Principal Component Analysis (PCA) and a regression model.

% Input:
%   timeWindowIndex : Index of the current time window for analysis.
%   timeDivision : Array indicating the division of time into bins.
%   labels : Array of direction labels corresponding to each trial.
%   directionIndex : Index indicating the current direction of movement being analyzed.
%   neuraldata : Matrix of neural firing rates, potentially filtered by previous steps.
%   pca_dimension : Number of principal components to retain in the PCA.
%   Interval : Array of time intervals for analysis.
%   currentXPositions : Matrix of x-coordinates of hand positions across trials.
%   currentYPositions : Matrix of y-coordinates of hand positions across trials.


% Output:
%   regressionCoefficientsX : Regression coefficients for predicting x-coordinates of hand positions.
%   regressionCoefficientsY : Regression coefficients for predicting y-coordinates of hand positions.
%   FilteredFiring : Neural data filtered by time and direction, used for regression.

    
    % Center the positions for the current time window

    centeredX = bsxfun(@minus, currentXPositions(:, timeWindowIndex), mean(currentXPositions(:, timeWindowIndex)));
    centeredY = bsxfun(@minus, currentYPositions(:, timeWindowIndex), mean(currentYPositions(:, timeWindowIndex)));
    
    % Filter firing data based on time and direction
    FilteredFiring = filterFiringData(neuraldata, timeDivision, Interval(timeWindowIndex), labels, directionIndex);

    % Center the firing data by subtracting the mean of each neuron's firing rate
    centeredWindowFiring = FilteredFiring  - mean(FilteredFiring ,1);
    
    % Perform PCA on the centered firing data to reduce dimensionality
    [principalVectors] = getPCA(centeredWindowFiring);
    principalComponents = principalVectors(:, 1:pca_dimension)' * centeredWindowFiring;

    % Calculate regression coefficients for X and Y using the regression matrix
    regressionMatrix = (principalComponents * principalComponents') \ principalComponents;
    regressionCoefficientsX = principalVectors(:, 1:pca_dimension) * regressionMatrix * centeredX;
    regressionCoefficientsY = principalVectors(:, 1:pca_dimension) * regressionMatrix * centeredY;

end

function FilteredFiring = filterFiringData(neuraldata, timeDivision, interval, labels, directionIndex)

% This function filters neural firing data based on specified time and direction criteria. 
% It first selects the firing data up to a given time point (interval) and then further 
% filters the data for a specific movement direction. The function finally centers the 
% filtered data by subtracting the mean firing rate across the selected trials for the specific direction.

% Inputs:
%   neuraldata : Matrix of neural firing rates
%   timeDivision : Array that maps each row in neuraldata to a time interval.
%   interval : Scalar specifying the time point up to which the data should be filtered. 
%   labels : Array of labels indicating the direction of movement associated with each column 
%            in 'neuraldata'. This is used to filter the data based on movement direction.
%   directionIndex : Scalar specifying the direction of movement to filter by. Only data columns
%                    (trials) that correspond to this movement direction will be selected.

% Output:
%   FilteredFiring - The resulting matrix of filtered neural firing rates, where the data has been 
%                    filtered to include only the time points up to 'interval' and trials that 
%                    correspond to the specified 'directionIndex'.


    % Filter the neural data to include only time points up to 'interval'
    timeFilter = timeDivision <= interval;
    % Further filter the data to include only trials corresponding to 'directionIndex'
    directionFilter = labels == directionIndex;
    FilteredFiring  = neuraldata(timeFilter, :);
    % Center the filtered data by subtracting the mean firing rate across the selected trials
    % for the specific direction. 
    FilteredFiring  = FilteredFiring (:, directionFilter) - mean(FilteredFiring(:, directionFilter), 1);
end
function [x,y,modelParameters, predicted]= positionEstimator(testData, modelParameters)

    %%%%%% Initialise parameters %%%%%%
    
    start = 320;
    finish = 560;
    group = 20;
    ema_decay = 0.3; % value chosen through multiple test trials
    
    %%%%% 1. Preprocess the trial data %%%%%
    %This step ensures that 
    % the data is in a suitable format for classification and regression.
    
    trialFinal = process_and_smooth(testData, group, ema_decay);            
    
    %%%%%% 2.  Determine indexer based on T_end %%%%%
    % Determine the indexer, which is used to select the appropriate model 
    % parameters for classification and regression based on the current time point 
    % (T_end) in the trial. This allows for dynamic adaptation as more data becomes 
    % available during a trial.
    
    T_end = size(testData.spikes, 2);
    indexer = min(max(floor((T_end - start) / group) + 1, 1), length(modelParameters.classify));
    
    %%%%%% 3. Extract firing data for PCA and PCR %%%%%%
    neuraldata = reshape(trialFinal.rates, [], 1);
    
    %%%%%% 4. Use Naive Bayes to classify direction  %%%%%
    % Classification of movement direction using a Naive Bayes classifier. If the current 
    % time point is within the specified range, classification is performed. 
    % Otherwise, the last known direction label is used.
    
    if T_end <= finish
    % Classification is applicable within the time range
     train_weight = modelParameters.classify(indexer).wTrain;
     meanFiringTrain = modelParameters.classify(indexer).mFiring;
     test_weight = modelParameters.classify(indexer).wTest' * (neuraldata - meanFiringTrain);
     naiveBayesModel = modelParameters.classify(indexer).naiveBayesModel;
     [outLabel, score] = predict(naiveBayesModel, test_weight');
    else
       
    % Beyond maxTime, use the last known label without re-classification
     outLabel = modelParameters.actualLabel;
    end
    modelParameters.actualLabel = outLabel; % Update the actual label in model parameters
    
    %%%%%%% 5. Estimate position using PCR results for both within and
    %%%%%%% beyond maxTime %%%%%

    % Estimate the hand position using PCR, applicable for both within the specified 
    % time range and beyond. This step uses regression coefficients and average firing 
    % rates to calculate the X and Y coordinates of the hand position.
    
    avX = modelParameters.averages(indexer).avX(:,outLabel);
    avY =  modelParameters.averages(indexer).avY(:,outLabel);
    meanFiring = modelParameters.pcr(outLabel, indexer).fMean;
    bx = modelParameters.pcr(outLabel,indexer).bx;
    by = modelParameters.pcr(outLabel,indexer).by;
    
    x = calculatePosition(neuraldata, meanFiring, bx, avX);
    y = calculatePosition(neuraldata, meanFiring, by, avY);
    predicted = outLabel; %The predicted direction of movement


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  HELPER FUNCTIONS   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function processed_data = process_and_smooth(trial, group, ema_decay)
% Processes trial data by re-binning, sqrt transformation, and smoothing firing rates with 
% an exponential moving average (EMA).
%
% Input: 
%  trial : input struct with spikes and handPos data.
%  group : new binning resolution in ms.
%  ema_decay : decay factor for EMA, controlling the degree of smoothing.

    % Initialize the output structure
    processed_data = struct;
    
    % Loop through each trial and reaching angle
    for i = 1:size(trial, 2)
        for j = 1:size(trial, 1)

            % Re-bin spikes
            spikes_total = trial(j, i).spikes;
            [neurons, points] = size(spikes_total);
            t_new = 1:group:points + 1;
            spikes_new = zeros(neurons, numel(t_new) - 1);

            for k = 1:length(t_new) - 1
                if k == length(t_new) - 1
                    spikes_new(:, k) = sum(spikes_total(:, t_new(k):end), 2); % Ensure we include all remaining data in the last bin
                else
                    spikes_new(:, k) = sum(spikes_total(:, t_new(k):t_new(k + 1) - 1), 2);
                end
            end

            % Apply sqrt transformation 
            spikes_new = sqrt(spikes_new);
            
            % Apply EMA smoothing
            ema_rates = zeros(size(spikes_new));
            for n = 1: neurons
                for t = 2:size(spikes_new, 2)
                    ema_rates(n, t) = ema_decay * spikes_new(n, t) + (1 - ema_decay) * ema_rates(n, t - 1);
                end
            end

            % Store processed data in the output structure
            processed_data(j, i).rates = ema_rates / (group / 1000); % Adjust rates to spikes per second
        end
    end
end


% Helper function for position calculation

function pos = calculatePosition(neuraldata, meanFiring, b, av)
    pos = (neuraldata(1:length(b)) - mean(meanFiring))' * b + av;
    pos = adjustFinalPosition(pos, T_end);
end

% Adjusting function for the final position

function pos = adjustFinalPosition(pos, T_finish)
    try
        pos = pos(T_finish, 1);
    catch
        pos = pos(end, 1); % Fallback to last position if specific T_end is not accessible
    end
end

end