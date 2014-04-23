function [trainingSet trainingLabels validationSet validationLabels] = createCrossValidationSet(tr_data, tr_labels, size_valid_set)

%Take a training set and create a validation set of size
%Return the training set and validation set
        

[n m] = size(tr_data);
split = floor(m/size_valid_set);
validationSet = tr_data(:,split:split:end);
validationLabels = tr_labels(split:split:end);

trainingSet = tr_data;
trainingLabels = tr_labels;

trainingSet(:,split:split:end) = [];
trainingLabels(split:split:end) = [];

	

