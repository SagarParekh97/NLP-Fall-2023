Run the following command to create a conda environment with the necessary packages

# create conda env
```conda env create -f env.yml```

# activate the env
```conda activate nlp_hw_906466769```

# Run
The best-performing models are provided in the ```hw3_model_906466769.zip``` file. Extract the zip file and move the ```save_data``` folder inside the main direcory. Run the python notebook ```hw3_906466769.ipynb```. The code will load the models from the folder.

In order to train the network again, simply delete the saved model and run the python notebook. In order to resume the training, change the variable ``` resume = True ```.

The output of the python notebook is also provided as a pdf, named ```hw3_906466769_Output.pdf```.

The approach, hyperparameters, and results are reported in the ```hw3_906466769_Report.pdf```.