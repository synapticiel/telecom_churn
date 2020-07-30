# Churn in telecom

With the rapid development of telecommunication industry, the service providers are inclined more towards expansion of the subscriber base. To meet the need of surviving in the competitive environment, the retention of existing customers has become a huge challenge. In the survey done in the Telecom industry, it is stated that the cost of acquiring a new customer is far more that retaining the existing one. Therefore, by collecting knowledge from the telecom industries can help in predicting the association of the customers as whether or not they will leave the company. The required action needs to be undertaken by the telecom industries in order to initiate the acquisition of their associated customers for making their market value stagnant.The goal of this article is to apply analytical techniques, using Elastic Stack, to predict a customer churn and analyse the churning and non churning customers.
This article has been inspired from [Elastic Blog](https://www.elastic.co/blog/using-elastic-supervised-machine-learning-for-binary-classification)


# Why supervised learning ? 

Supervised machine learning trains a model from labeled data (means a dataset with a list of customers with a label to show if the customer has churned or not). This is in contrast to unsupervised learning where the model learns from unlabeled data. In [data frame analytics](https://www.elastic.co/guide/en/machine-learning/master/ml-dfa-overview.html), Elastic Stack give you the toolset to create supervised models without knowledge of the underlying algorithms. You can then use the trained models to make predictions on unobserved data (via inference). The model applies what it learned through training to provide a prediction.

Why use this instead of [anomaly detection](https://www.elastic.co/guide/en/machine-learning/master/xpack-ml.html) ? Anomaly detection, an example of unsupervised learning, is excellent at learning patterns and predicting metrics based on time series data. This is powerful for many use cases, but what if we wanted to identify bad actors by detecting suspicious domains ? What about automatically detecting language and applying the appropriate analyzers for better search ? 

These use cases need richer feature sets to provide accurate predictions. They also have known examples from which to learn. Meaning, there exists a training dataset where the data is labeled with the correct prediction. Transforms supplies the tools needed to build complex feature sets. Data frame analytics can train supervised machine learning models. We can use both to deploy the feature generation and supervised models for production use on streaming data.

# Supervised learning example : binary classification

Let’s build and use a binary classification model from scratch. Don’t be intimidated. No data science degree is necessary. We will use the building blocks within the Elastic Stack to create our feature set, train our model, and deploy it in an ingest pipeline to perform inference. Using an ingest processor allows our data to be enriched with the model’s predictions before being indexed.

In this example, we will predict whether a telecom customer is likely to churn based on features of their customer metadata and call logs. We’ll use examples of churned and not churned customers as the training dataset. 

# Setting up your environment

To follow along, start a free trial of [Elastic Cloud](https://www.elastic.co/elasticsearch/service), spin up a [new deployment](https://www.elastic.co/guide/en/cloud/current/ec-create-deployment.html) or simply download Elasticsearch, Kibana and Logstash from [Elastic](https://www.elastic.co) web site, download the [calls.csv](./call.7z) and [customers.csv](./customers.7z) files, and upload them via the [CSV upload feature](https://www.elastic.co/blog/importing-csv-and-log-data-into-elasticsearch-with-file-data-visualizer) (or use Logstash). Data is derived from a dataset referenced in various sources : [openml](https://www.openml.org/d/40701), [kaggle](https://www.kaggle.com/becksddf/churn-in-telecoms-dataset), [Larose 2014](http://dataminingconsultant.com/DKD2e_data_sets.zip). This dataset was disaggregated into line items, and random features were added. Phone numbers and other data are fabricated and any resemblance to real data is coincidental. 

Once uploaded you should have data that looks like this :

In the customers index :

```json
{
  "record_id": 332,
  "@timestamp": "2020-09-08T10:16:15.019Z",
  "phone_number": "2253803437",
  "account_age": 86,
  "number_vmail_messages": 0,
  "customer_service_calls": 7,
  "voice_mail_plan": "no",
  "state": "AB",
  "international_plan": "no",
  "churn": 1,
}
```
In the calls index :

```json
{
  "record_id": 1,
  "@timestamp": "2020-07-07T07:21:40.102Z",
  "phone_number": "2253803437",
  "dialled_number": "2253283266",
  "call_charges": 0.34497474848639376,
  "call_duration": 3.3484235812459193
}
```

# Building the feature set

First, we want to build out our feature set. We have two sources of data: customer metadata and call logs. This requires the use of transforms and the enrich processor. This combination allows us to merge the metadata along with the call information in the other index.

The enrichment policy and pipeline. (Execute the commands in the Kibana dev console)

````json
# Enrichment policy for customer metadata
PUT /_enrich/policy/customer_metadata
{
  "match": {
    "indices": "customers",
    "match_field": "phone_number",
    "enrich_fields": [
      "account_age",
      "churn",
      "customer_service_calls",
      "international_plan",
      "number_vmail_messages",
      "state",
      "voice_mail_plan"
    ]
  }
}
````
````json
# Execute the policy so we can populate with the metadata
POST /_enrich/policy/customer_metadata/_execute
````

```json
# Our enrichment pipeline for generating features
PUT _ingest/pipeline/customer_metadata
{
  "description": "Adds metadata about customers by lookup on phone_number",
  "processors": [
    {
      "enrich": {
        "policy_name": "customer_metadata",
        "field": "phone_number",
        "target_field": "customer",
        "max_matches": 1
      }
    }
  ]
}
```

We can now build a transform that utilizes our new pipeline. Here is the transform definition:

```json
# transform for enriching the data for training
PUT _transform/customer_churn_transform
{
  "source": {
    "index": [
      "calls"
    ]
  },
  "dest": {
    "index": "churn_transform_index",
    "pipeline": "customer_metadata"
  },
  "pivot": {
    "group_by": {
      "phone_number": {
        "terms": {
          "field": "phone_number"
        }
      }
    },
    "aggregations": {
      "call_charges": {
        "sum": {
          "field": "call_charges"
        }
      },
      "call_duration": {
        "sum": {
          "field": "call_duration"
        }
      },
      "call_count": {
        "value_count": {
          "field": "dialled_number"
        }
      }
    }
  }
}
```

Now that the transform is created, we can start it and see its progress on the Transforms page under Stack Management.

<img src="./screens/tranform.png" align="middle">


Once tranfor executed  you should have data that looks like this :

In the customers churn_transform_index :

```json
{
  "phone_number": "2253271058",
  "call_count": 253,
  "call_charges": 45.72,
  "call_duration": 488.5,
  "customer": {
    "voice_mail_plan": "no",
    "number_vmail_messages": 0,
    "churn": 0,
    "account_age": 112,
    "phone_number": "2253271058",
    "state": "AB",
    "international_plan": "no",
    "customer_service_calls": 0
  }
}
```
