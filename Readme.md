
## Finding Research Involving Animal Ethics

## Goal: try to pick up all animal-ethic related publications for funding

## Procedure:

### Build vocabulary
[Build Vocabulary.R](https://github.com/lucas3359/text-mining-AnimalEthics/blob/master/Build%20Vocabulary.R)

1. Using key words that are confirmed as animal ethics related to build initial vocabulary
2. Use algorithm to pick up those passages containing the key words (filtering only one faculty first, with other faculties, publications containing the words are not necessarily animal ethics related )
3. Assign category "Animal Related" with values Y to those that have been picked, and N to those are not
4. Train a classification model based on the assigned category data(excluding 2011 results)
5. Predict the 2011 results and evaluate the model using a set of validated faculty's publications to validate classification model (validated publications are from Year 2011)
6. Manually check validated publications provided by the faculty and the false positives from the model, to generate a list of keywords not yet in the model
7. Expand the vocabulary for the model using the results from step 6
8. Repeat steps 2-7 using a different year for validation to build vocabulary

### Build model
1. Merge keywords from all vocabularies for each year
2. Assign category "Animal Related" with values Y to those that have been picked up by completed vocabulary, and N to those are not
3. Train model using category

### Predict
1. If accuracy is not acceptable for use with other faculties, rebuild the vocabulary using data from that faculty.