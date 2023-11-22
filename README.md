# Question 2: Programming

This is the project for Question 2: Programming.

## Project Structure

The project structure is as follows:

- `raw_data`: contains the raw data files downloaded from the link provided in the instruction.
- `pt_data`: contains the .pt files generated from the raw data files. The generation process is described in the `integration.ipynb` notebook.
- `environment.yml`: contains the conda environment specification.
- `integration.ipynb`: contains the data integration steps.
- `train.py`: contains the training script.

## Reproducibility

To reproduce the results, please install the conda environment from the `environment.yml` file by running the following command:

```bash
conda env create -f environment.yml
```

Then, activate the environment:

```bash
conda activate UvA-Project
```

Finally, run the `train.py` script:

```bash
python train.py
```

You can also run the `integration.ipynb` notebook to reproduce the data integration steps.

I've validated the above steps on a Windows 11 machine with conda 23.1.0 installed to make sure the environment is reproducible.

### Data Integration Highlights

I've added detailed comments in the `integration.ipynb` notebook to explain the data integration steps. I would like to highlight the following points:
- I use pandas, sklearn, and pytorch to implement the data integration steps.
- When I loaded the data, I found that `reviews-data-2.tsv` had a different column order, so I handled it separately.
- After checking the unique values of the `id` column in the two data files, I am sure that I can merge the two data files by the `id` column directly.
- The `category` column in the product data file contains some typos. The `kitchen` is spelled as `Kithen`. I've fixed the typos.
- I encoded text features using `TF-IDF`.
- I encoded the `rating` column using one-hot encoding.
- I splited the data into train, validation, and test sets with a ratio of 6:2:2. The random seed is set to 811.
- I plotted the distribution of the `rating` column and `category` column in each set to make sure the data is split properly.
- I saved the processed data into `.pt` files for later use.
- The training results can be seen at the end of the notebook. You can also run the `train.py` script to reproduce the results.