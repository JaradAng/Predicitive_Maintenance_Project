# Prediciting Machine Failure

--- 

## Project Goal 

- The goal of this project is to create a reproducible  machine learning classification model to predict the whether or not a machine will fail, as well as what type of failure may occur 
- In order to achieve an accurate prediction, the project will identify key drivers of failures. This notebook serves as a way to understand why and how I made the prediction model.

---

## Project Description

-  Time is money during manufacturing. By prediciting whether or not a machine will fail and the type of failure, we are able to find the conditions which caused the failure. Predicting failure will eliminate downtime to repair broken equipment saving money from the machine parts and the labor required to do the repairs. This will help companies produce more good with less wasted capital, improving overall the overall bottom line for the company.

---

### Key Questions

1. How does RPM affect machine failure?

2. How does toque affect machine failure?

3. How does processing temperatures affect machine failures?

4. Does tool wear affect machine failure?

5. Does air temperature affect machine failure?


---

### How to Replicate the Results


- Clone my repo (including the proj_acquire.py)
- Libraries used are pandas, matplotlib, seaborn, numpy, sklearn, imblearn, xgboost, and tpot
- You should be able to run final_maintenance_notebook.ipynb

---

### The plan

1. Wrangle the data using the proj_acquire.py folder     
2. Additionally inside of the proj_acquire.py folder, I prepped, cleaned, and split the dataset.  I also featured engineered columns to help the exploration visualizations. 
3. During the exploration phase in maintenance_explore.ipynb, I visualize multiple features to asses which features to include in the model and to find if the features are statistically significant. Inside my explore notebook, I make multiple models and compare the results against each other to determine the final models to include within the final report.
4. Move and organize all important and useful features to a final notebook to deliver a streamlined product for stakeholders and management.  
5. Deliver a final report notebook with all required .py files

---

### Data Dictionary

Variable | Definiton | 
--- | --- | 
RPM | Revolutions per minute |
--- | --- | 
Torque | Measure of force to produce rotational speed |
--- | --- | 
air_temp | Air temperature |
--- | --- | 
Process_temp | The temperature of machine during processing |
--- | --- | 
tool_wear | The amount of wear the tool has  |
--- | --- | 
Type | H = High, M = Medium, L = Low | 
--- | --- | 
machine_failure | binary outcome on whether or not the machine experienced failure |
--- | --- | 
Failure_type | Heat dissipation failure, tool wear failure, random failure, power failure, overstrain failure, no failure|
--- | --- | 




### Exploring the Questions and Hypothesis


1. How does RPM affect machine failure?
- The chart shows that lower rpms experience the most failures by a wide margin.

Ho - Rpm is independent from machine failure
Ha - Rpm is dependent on machine failure

Result: We reject the null.

2. How does toque affect machine failure?
- Torque appears to be opposite of rpm and experiences more failure on the high end of the range. Again the failure is by a large margin

Ho - Machine failure is independent of torque
Ha - Machine failure is dependent on torque

Result: We reject the null.

3. How does processing temperatures affect machine failures?
There is a difference in failure rates amongst the bins. This has a wider band of failure with only the lower temp range no experiencing large number of failures

Ho - Machine failure is indepenent of process temps
Ha - Machine failure is dependent on process temps

Result: We reject the null.

4. Does tool wear affect machine failure?
- The older tools with more wear show much higher failure rates than those not as old. 

Ho - Machine failure is independent of tool wear
Ha- Machine failure is dependent of tool wear

Result: We reject the null.


5. Does air temperature affect machine failure?
- Machine failures appear to happen in the mild to hot temperature zones. Cool temps has less affect on failures

Ho - Machine failure is independent on air temps
Ha - Machine failure is dependent on air temps

Result: We reject the null. 







## Exploration Summary
- Each of the variables explored show statistical significance. 
- Air and process temperatures show the most variety within overall failures. 
- Rpm shows failure in the low range of values
- Tool wear and Torque show failure in the higher ranges compared to rpm.
- The multicolinearity is ok becuase of the use of classification models. I explore handing multicolinearity in different explore notebooks
- The bins were dropped as I explored those as well in different notebook and selected RFE did not choose a single bin.
- I used anova tests for each of the bins becuase there were three bins each
---
## Modeling phase inwhich the first model will be an overal predictor of if a machine will fail or not fail. 
- I will be using recall score to determine the how accurately the model find true positives or those that failed and were predicted to fail with accuracy as secondary measure
- The first phase to determine the binary outcome of whether the machine will fail or not fail
- The second model will determine the type of failure the machine occured. 
--- 

# Summary
* The best classification model for fail or not fail is the Decision Tree Classifier. The modele did better on the train due. Decision tree had an 81% recall rate which means guessed actual failures correctly and an overall accuracy of 86%.
* The top two performing models for both the multiclassification and regular classification were Decision Tree and XGBoost Classifier. The decision tree performed slight better in over recall with a 73% compared to XGBoosts 66%. However XG has a 97% accuracy compared to just 7% of decisions tree, due to the extreme difference in accuracy I will be going with XGBoost for the test. 
* Recall proved to be lower than overall accuracy. However, if I were to use accuracy for the baseline, I would be able to beat baseline by roughly 2%
* After running the models on test they slightly beat validate. Since there is no recall for baseline I was not able to test against baseline

---
# Recommendations
* Gain additional information from the machines such as time, so that we can predict when a tool might fail 
* Try to use a regression for the binary classification to see how well that might perform.
---
# Next Steps
* Build a model pipeline to predict if it would fail and then what type of failure more accurately
* Work if machines to find more features to add to model
* Proof of concecpt - can continue to build and work on the model. 