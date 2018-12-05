# Logistic_Regression_Model
This piece of code enable you to extract flight information search terms from natural language using logistic regression model. You could run the code using following command:

$ python tagger.py [args...]

Where [args...] is a placeholder for 8 command-line arguments:

* <train_input> : path to training input .tsv file
* <validation_input> : path to validation input .tsv file
* <test_input> : path to test input .tsv file
* <train_out> : path to output .labels file to which the prediction on training data will be written
* <test_out> : path to output .labels file to which the prediction on test data should be written
* <metrics_out>: path of output .txt file to which training error and test error will be written
* <num_epoch> : positive integer specifying the number of ties SGD loops through all of the training data
* <featrue_flag> : integer taking value 1 or 2 specifying whether to construct Model 1 (feature only contains current word) or Model 2 (feature contains previous word, current word and next word).
