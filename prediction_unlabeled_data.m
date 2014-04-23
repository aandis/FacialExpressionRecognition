clear all
%% Load Data
load labeled_data.mat;
load val_images.mat;
load unlabeled_images.mat;

ntr = size(tr_images, 3);
ntest = size(val_images,3);
nulab = size(unlabeled_images,3);
h = size(tr_images,1);
w = size(tr_images,2);
tr_images = double(reshape(tr_images, [h*w, ntr]));
valid_images = double(reshape(val_images, [h*w, ntest]));
unlabeled_images = double(reshape(unlabeled_images, [h*w, nulab]));
unlabeled_images = unlabeled_images(:,1:30000);
inputs_train = tr_images;
targ_train = tr_labels;
inputs_valid = inputs_train;
targ_valid = targ_train;

inputs_test = unlabeled_images;

%[inputs_train targ_train inputs_valid targ_valid] = createCrossValidationSet(inputs_train, targ_train, 200);


% Subtract mean for each image
tr_mu = mean(inputs_train);
valid_mu = mean(inputs_valid);
test_mu = mean(inputs_test);
inputs_train = bsxfun(@minus, inputs_train, tr_mu);
inputs_valid = bsxfun(@minus, inputs_valid, valid_mu);
inputs_test = bsxfun(@minus, inputs_test, test_mu);



% Normalize variance for each image
tr_sd = var(inputs_train);
tr_sd = tr_sd + 0.01; % for extreme cases
tr_sd = sqrt(tr_sd);
inputs_train = bsxfun(@rdivide, inputs_train, tr_sd);  

valid_sd = var(inputs_valid);
valid_sd = valid_sd + 0.01; % for extreme cases
valid_sd = sqrt(valid_sd);
inputs_valid = bsxfun(@rdivide, inputs_valid, valid_sd);

test_sd = var(inputs_test);
test_sd = test_sd + 0.01;
test_sd = sqrt(test_sd);
inputs_test = bsxfun(@rdivide, inputs_test, test_sd);  


num_classes = length(unique(targ_train));


temp = zeros(num_classes,length(targ_train));
for t = 1:length(targ_train)
	temp(targ_train(t),t) = 1;
end
target_train = temp;
temp = zeros(num_classes,length(targ_valid));
for t = 1:length(targ_valid)
	temp(targ_valid(t),t) = 1;
end
target_valid = temp;

init_nn;
for count = [1:10]
	train_nn
end
test_nn;



[hypothesis prediction] = max(prediction);
ulab_labels = prediction(hypothesis>0.995);
unlab_images = inputs_test(:,hypothesis>0.995);

save('unlab_images.mat', 'unlab_images', 'ulab_labels');



%%% Fill in the test labels with 0 if necessary
%if (length(prediction) < 1253)
%  prediction = [prediction'; zeros(1253-length(prediction), 1)];
%end


%%% Print the predictions to file
%fprintf('writing the output to prediction.csv\n');
%fid = fopen('prediction.csv', 'w');
%fprintf(fid,'%s,%s\n', 'Id','Prediction');
%for i=1:length(prediction)
%  fprintf(fid,'%d,%d\n', i,prediction(i));
%end
%fclose(fid);


