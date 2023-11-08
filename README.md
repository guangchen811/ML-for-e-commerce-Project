# uva_application_project

## Instruction
we ask you to implement an ML pipeline for a fictitious e-commerce use case. We provide a small data set from an e-commerce platform, which contains products from two categories: Kitchen and Jewelry. The dataset contains two kinds of input files with separated data (available at [https://surfdrive.surf.nl/files/index.php/s/LDwpIdG7HHkQiOs](https://surfdrive.surf.nl/files/index.php/s/LDwpIdG7HHkQiOs) )

Product data (id, category, product title), e.g.:
daa54754-af9c-41c0-b542-fe5eabc5919c Kitchen Bodum French Press Coffeemaker
Reviews (id, rating, review_text), e.g.: 
daa54754-af9c-41c0-b542-fe5eabc5919c 5 Great!

We ask you to train and evaluate a classifier that predicts the category from the product title, rating and review text. Please note that this task is not meant to be a “kaggle contest”, instead it is meant to evaluate your software engineering skills (not to develop the best-performing classifier for the task). We therefore ask you to use a logistic regression model and encode the features in a simple way. Please focus your efforts on the data integration steps, the readability of your code and the reproducibility and practicality of your solution. Please provide us with the code of your solution via a link to an online repository. You are free to choose any programming language, data processing engine and ML framework.