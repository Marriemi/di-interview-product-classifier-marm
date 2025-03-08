# DI Interview: Product classifier 

This repository hosts all the necessary elements to complete the following DI exercise:

1. Create a text model to classify products based on their title
2. Use this model in a REST API, developed in Python

## 1 - Train the model

Use the Jupyter Notebook `training/training.ipynb`. 
The training set is located in BigQuery.
You'll find the necessary starting code to be able to retrieve it in the notebook.

The referenced `credentials.json` file will be provided by email by our team and should be put in the `training` folder

**Goal**: train a model based on the titles and subtitles in the set to match a product type.

## 2 - Build the classifier API

Once the model trained, use the `app` folder to write the code for the REST API using this model.

The API should meet the following contract:

```bash
curl -XPOST 'http://localhost:5000/classify' \
-H 'Content-Type: application/json' \
-d '{ "title": "my product title containing aquarelle" }'
```

and the response should look like this: 

```json
{
    "title": "my product title containing aquarelle",
    "top_3_results": [
        {
            "product_type": "painting",
            "score": 0.xxxx
        },
        {
            "product_type": "aquarium",
            "score": 0.xxxx
        },
        {
            "product_type": "diving_suit",
            "score": 0.xxxx
        }
    ],
    "product_type": "painting"
}
```

No library is imposed to write the API, choose the one you want.

## An issue, a question?

Send us a mail at [gm-data@swissmarketplace.group](mailto:gm-data@swissmarketplace.group)
