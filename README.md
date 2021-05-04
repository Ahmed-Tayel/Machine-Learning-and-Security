# Machine Learning and Security
This case study provides a solution to a security problem by using a machine learning model, the problem is mainly on any files containing encryption code that could harm the receiver’s machine and data.

Machine learning models have been developed to identify if the file contains any encryption function by examining the assembly code of the file, a dataset has been
collected containing sets of assembly commands (as inputs) and every set of the commands are labeled by the type of the function, whether it’s ‘Math’, ‘Sorting’,’String’,
or ‘Encryption’.

Each of the four classes has a pattern in its assembly language, for example: using a lot of XOR operations in encryption assembly, while using a lot of swaps in sorting, etc, by extracting these patterns in someway and by training the appropriate model, the model could be able to classify the functions with good performance.

The case study compared different solutions with respect to the overall accuracy and encryption function precision, as it’s more important for the model to correctly classify the encryption function than any other function, in other words, to minimize the wrong classification of the encryption function in the first place.

The difference between normalized and not normalized data is reported, tuning “Support Vector Machine” (SVM) polynomial degree kernel, and comparing SVM with “Gaussian
Naive Bayes” by stressing the two models with noise.

## NOTE
the dataset file is not full to satisfy github storage policy (40% of the full dataset), if you need the full dataset please contact me I'll send it to you