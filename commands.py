from ludwig.train_parquet import create_preprocessed_parquet, train_parquet

"""
model def:
input_features:
    -
        name: text
        type: text
        level: word
output_features:
    -
        name: intent
        type: category
training:
    epochs: 2
    
    
data:
text,intent,counts,locale
yes yes yes yes,accept,5,en
beautiful,other,1,en
beautiful beautiful,other,1,en
okay now eggmania that I will do 1546 Oak Tree Road in Woodbridge Township,other,1,en
Beach Fair,other,1,en
Beach Fair please,other,1,en
all right,accept,1,en
where's Abigail from,other,1,en
like I'm doing,other,1,en

"""

create_preprocessed_parquet(
    None,
    'dispatch_bot/model_definition.yaml',
    data_train_csv='dispatch_bot/data/train.csv',
    data_validation_csv='dispatch_bot/data/validation.csv',
    data_test_csv='dispatch_bot/data/test.csv'
)

_ = train_parquet(
    {},
    model_definition_file='dispatch_bot/model_definition.yaml',
    train_parquet_path='dispatch_bot/data/train.parquet/',
    validation_parquet_path='dispatch_bot/data/validation.parquet/',
    test_parquet_path='dispatch_bot/data/test.parquet/',
    train_set_metadata_json='dispatch_bot/data/train.json'
)
