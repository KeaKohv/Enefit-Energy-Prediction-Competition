from typing import Iterable
from typing import Optional

import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import as_stacked_batches
from gluonts.itertools import Cyclic, Cached
from gluonts.time_feature import (
    time_features_from_frequency_str,
    get_lags_for_frequency,
)
import gluonts.transform as T
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    RemoveFields,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
    RenameFields,
    CausalMeanValueImputation,
    DummyValueImputation,
    MeanValueImputation,
)
from gluonts.transform.sampler import InstanceSampler
from transformers import PretrainedConfig
from transformers import TimeSeriesTransformerConfig

from functools import partial
from gluonts.time_feature import time_features_from_frequency_str
from transformers import TimeSeriesTransformerForPrediction
from accelerate import Accelerator
from torch.optim import AdamW

from functools import lru_cache

@lru_cache(10_000)
def convert_to_pandas_period(date, freq):
    return pd.Period(date, freq)


def transform_start_field(batch, freq):
    batch["start"] = [convert_to_pandas_period(date, freq) for date in batch["start"]]
    return batch


def create_transformation(freq: str, config: PretrainedConfig) -> Transformation:
    # create a list of fields to remove later
    remove_field_names = []
    if config.num_static_real_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_REAL)
    if config.num_dynamic_real_features == 0:
        remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
    if config.num_static_categorical_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_CAT)

    return Chain(
        # step 1: remove static/dynamic fields if not specified
        [RemoveFields(field_names=remove_field_names)]
        # step 2: convert the data to NumPy (potentially not needed)
        + (
            [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT,
                    expected_ndim=1,
                    dtype=int,
                )
            ]
            if config.num_static_categorical_features > 0
            else []
        )
        + (
            [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL,
                    expected_ndim=1,
                )
            ]
            if config.num_static_real_features > 0
            else []
        )
        + (
            [
                AsNumpyArray(
                    field=FieldName.FEAT_DYNAMIC_REAL,
                    expected_ndim=2,
                )
            ]
            if config.num_dynamic_real_features > 0
            else []
        )

        + [
            AsNumpyArray(
                field=FieldName.TARGET,
                # we expect an extra dim for the multivariate case:
                expected_ndim=1 if config.input_size == 1 else 2,
            ),
            # step 3: handle the NaN's by filling in the target with zero
            # and return the mask (which is in the observed values)
            # true for observed values, false for nan's
            # the decoder uses this mask (no loss is incurred for unobserved values)
            # see loss_weights inside the xxxForPrediction model
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            # AddObservedValuesIndicator(
            #     target_field=FieldName.FEAT_DYNAMIC_REAL,
            #     output_field=FieldName.FEAT_DYNAMIC_REAL,
            #     imputation_method= MeanValueImputation()
            # ),
            # step 4: add temporal features based on freq of the dataset
            # these serve as positional encodings
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                time_features=time_features_from_frequency_str(freq),
                pred_length=config.prediction_length,
            ),
            # step 5: add another temporal feature (just a single number)
            # tells the model where in the life the value of the time series is
            # sort of running counter
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=config.prediction_length,
                log_scale=True,
            ),
            # step 6: vertically stack all the temporal features into the key FEAT_TIME
            VstackFeatures(
                output_field=FieldName.FEAT_TIME,
                input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                + (
                    [FieldName.FEAT_DYNAMIC_REAL]
                    if config.num_dynamic_real_features > 0
                    else []
                ),
            ),
            # step 7: rename to match HuggingFace names
            RenameFields(
                mapping={
                    FieldName.FEAT_STATIC_CAT: "static_categorical_features",
                    FieldName.FEAT_STATIC_REAL: "static_real_features",
                    FieldName.FEAT_TIME: "time_features",
                    FieldName.TARGET: "values",
                    FieldName.OBSERVED_VALUES: "observed_mask",
                }
            ),
        ]
    )

def create_instance_splitter(
    config: PretrainedConfig,
    mode: str,
    train_sampler: Optional[InstanceSampler] = None,
    validation_sampler: Optional[InstanceSampler] = None,
) -> Transformation:
    assert mode in ["train", "validation", "test"]

    instance_sampler = {
        "train": train_sampler
        or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=config.prediction_length
        ),
        "validation": validation_sampler
        or ValidationSplitSampler(min_future=config.prediction_length),
        "test": TestSplitSampler(),
    }[mode]

    return InstanceSplitter(
        target_field="values",
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=instance_sampler,
        past_length=config.context_length + max(config.lags_sequence),
        future_length=config.prediction_length,
        time_series_fields=["time_features", "observed_mask"],
    )


def create_train_dataloader(
    config: PretrainedConfig,
    freq,
    data,
    batch_size: int,
    num_batches_per_epoch: int,
    shuffle_buffer_length: Optional[int] = None,
    cache_data: bool = True,
    **kwargs,
) -> Iterable:
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",

    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
        "future_values",
        "future_observed_mask",
    ]

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=True)
    if cache_data:
        transformed_data = Cached(transformed_data)

    # we initialize a Training instance
    instance_splitter = create_instance_splitter(config, "train")

    # the instance splitter will sample a window of
    # context length + lags + prediction length (from the 366 possible transformed time series)
    # randomly from within the target time series and return an iterator.
    stream = Cyclic(transformed_data).stream()
    training_instances = instance_splitter.apply(stream)

    return as_stacked_batches(
        training_instances,
        batch_size=batch_size,
        shuffle_buffer_length=shuffle_buffer_length,
        field_names=TRAINING_INPUT_NAMES,
        output_type=torch.tensor,
        num_batches_per_epoch=num_batches_per_epoch,
    )


def create_backtest_dataloader(
    config: PretrainedConfig,
    freq,
    data,
    batch_size: int,
    **kwargs,
):
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",

    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data)

    # We create a Validation Instance splitter which will sample the very last
    # context window seen during training only for the encoder.
    instance_sampler = create_instance_splitter(config, "validation")

    # we apply the transformations in train mode
    testing_instances = instance_sampler.apply(transformed_data, is_train=True)

    return as_stacked_batches(
        testing_instances,
        batch_size=batch_size,
        output_type=torch.tensor,
        field_names=PREDICTION_INPUT_NAMES,
    )


def create_test_dataloader(
    config: PretrainedConfig,
    freq,
    data,
    batch_size: int,
    **kwargs,
):
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=False)

    # We create a test Instance splitter to sample the very last
    # context window from the dataset provided.
    instance_sampler = create_instance_splitter(config, "test")

    # We apply the transformations in test mode
    testing_instances = instance_sampler.apply(transformed_data, is_train=False)

    return as_stacked_batches(
        testing_instances,
        batch_size=batch_size,
        output_type=torch.tensor,
        field_names=PREDICTION_INPUT_NAMES,
    )


# %%

df_train = pd.read_csv('./generated-train-nanless.csv', parse_dates=[1])
feat_dynamic_real_cols = "eic_count,installed_capacity,temperature,dewpoint,cloudcover_high,cloudcover_low,cloudcover_mid,cloudcover_total,10_metre_u_wind_component,10_metre_v_wind_component,direct_solar_radiation,surface_solar_radiation_downwards,snowfall,total_precipitation,temperature_forecast_local_0h,dewpoint_forecast_local_0h,cloudcover_high_forecast_local_0h,cloudcover_low_forecast_local_0h,cloudcover_mid_forecast_local_0h,cloudcover_total_forecast_local_0h,10_metre_u_wind_component_forecast_local_0h,10_metre_v_wind_component_forecast_local_0h,direct_solar_radiation_forecast_local_0h,surface_solar_radiation_downwards_forecast_local_0h,snowfall_forecast_local_0h,total_precipitation_forecast_local_0h,temperature_historical_0h,dewpoint_historical_0h,rain,snowfall_historical_0h,surface_pressure,cloudcover_total_historical_0h,cloudcover_low_historical_0h,cloudcover_mid_historical_0h,cloudcover_high_historical_0h,windspeed_10m,winddirection_10m,shortwave_radiation,direct_solar_radiation_historical_0h,diffuse_radiation,temperature_historical_local_0h,dewpoint_historical_local_0h,rain_historical_local_0h,snowfall_historical_local_0h,surface_pressure_historical_local_0h,cloudcover_total_historical_local_0h,cloudcover_low_historical_local_0h,cloudcover_mid_historical_local_0h,cloudcover_high_historical_local_0h,windspeed_10m_historical_local_0h,winddirection_10m_historical_local_0h,shortwave_radiation_historical_local_0h,direct_solar_radiation_historical_local_0h,diffuse_radiation_historical_local_0h".split(',')
feat_static_cat_columns = ["is_business", "product_type", "county", "is_consumption"]

prediction_length = 24*2
context_length = 24*7

n_validation = prediction_length
n_test = prediction_length

cardinality = [df_train[col].nunique() for col in feat_static_cat_columns]
num_static_categorical_features = len(feat_static_cat_columns)
embedding_dimension = [64] * len(cardinality)
num_dynamic_real_features = len(feat_dynamic_real_cols)

freq = "H"
time_features = time_features_from_frequency_str(freq)  # ? from the tutorial
lags_sequence = get_lags_for_frequency(freq) # list of integers for some goddamn lag features

config = TimeSeriesTransformerConfig(
    prediction_length=prediction_length,
    context_length=context_length,
    lags_sequence=lags_sequence,
    # we'll add 2 time features ("month of year" and "age", see further):
    num_time_features=len(time_features) + 1,
    num_dynamic_real_features=num_dynamic_real_features,
    num_static_categorical_features=num_static_categorical_features,
    cardinality=cardinality,
    embedding_dimension=embedding_dimension,
    # transformer params:
    encoder_layers=4,
    decoder_layers=4,
    d_model=32,
    distribution_output="student_t",
)

train = []
validation = []
test = []

item_id = 0

for col in feat_static_cat_columns:
    print(col, df_train[col].unique())

for (is_business, product_type, county, is_consumption), grp in df_train.groupby(
    ["is_business", "product_type", "county", "is_consumption"]
):
    assert grp.isna().values.sum() <= 0, grp.isna().sum() # NO NANS NO MORE

    # calculate last index for training data assuming we want 240 test examples 
    end = len(grp) - (prediction_length + context_length + 1 + 240)

    test_grp = grp.iloc[end:]
    train_grp  = grp.iloc[:end]

    start = train_grp["datetime"].min().to_pydatetime()
    target = train_grp["target"]

    feat_dynamic_real = train_grp[feat_dynamic_real_cols].values.T.astype(
        np.float32
    )
    feat_static_cat = [ is_business, product_type, county, is_consumption ]

    train.append(
        dict(
            start=start,
            target=target,
            feat_static_cat=feat_static_cat,
            feat_dynamic_real=feat_dynamic_real,
            item_id=f"{item_id}",
        )
    )
    item_id += 1

    for i in range(48):
        test_sample = test_grp.iloc[i+48:i+48+context_length+prediction_length]
        test.append(
            dict(
                start=test_sample['datetime'].min().to_pydatetime(),
                target=test_sample['target'],
                feat_static_cat=feat_static_cat,
                feat_dynamic_real=test_sample[feat_dynamic_real_cols].values.T.astype(np.float32),
                item_id=f"{item_id}",
            )
        )
        item_id += 1

my_dataset = datasets.DatasetDict(
    dict(
        train=datasets.Dataset.from_list(train),
        test=datasets.Dataset.from_list(test),
        validation=datasets.Dataset.from_list(validation),
    )
)

train_dataset = my_dataset["train"]
test_dataset = my_dataset["test"]

train_dataset.set_transform(partial(transform_start_field, freq=freq))
test_dataset.set_transform(partial(transform_start_field, freq=freq))

model = TimeSeriesTransformerForPrediction(config)

train_dataloader = create_train_dataloader(
    config=config,
    freq=freq,
    data=train_dataset,
    batch_size=64,
    num_batches_per_epoch=2000,
)

test_dataloader = create_backtest_dataloader(
    config=config,
    freq=freq,
    data=test_dataset,
    batch_size=100,
)

accelerator = Accelerator()
device = accelerator.device

model.to(device)
optimizer = AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=1e-6)

model, optimizer, train_dataloader = accelerator.prepare(
    model,
    optimizer,
    train_dataloader,
)

model.train()
for epoch in range(20):
    losses = []
    for idx, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        outputs = model(
            static_categorical_features=batch["static_categorical_features"].to(device)
            if config.num_static_categorical_features > 0
            else None,
            static_real_features=batch["static_real_features"].to(device)
            if config.num_static_real_features > 0
            else None,
            past_time_features=batch["past_time_features"].to(device),
            past_values=batch["past_values"].to(device),
            future_time_features=batch["future_time_features"].to(device),
            future_values=batch["future_values"].to(device),
            past_observed_mask=batch["past_observed_mask"].to(device),
            future_observed_mask=batch["future_observed_mask"].to(device),
        )
        loss = outputs.loss
        losses.append(loss.item())

        # Backpropagation
        accelerator.backward(loss)
        optimizer.step()

    print(f'{epoch=} {np.mean(losses)=}')

model.save_pretrained( save_directory='./vanilla_transformer')

model.eval()

forecasts = []

for batch in test_dataloader:
    outputs = model.generate(
        static_categorical_features=batch["static_categorical_features"].to(device)
        if config.num_static_categorical_features > 0
        else None,
        static_real_features=batch["static_real_features"].to(device)
        if config.num_static_real_features > 0
        else None,
        past_time_features=batch["past_time_features"].to(device),
        past_values=batch["past_values"].to(device),
        future_time_features=batch["future_time_features"].to(device),
        past_observed_mask=batch["past_observed_mask"].to(device),
    )
    print(outputs.sequences.shape)
    forecasts.append(outputs.sequences.cpu().numpy())

forecasts[0].shape

forecasts = np.vstack(forecasts)
print(forecasts.shape)

forecast_median = np.median(forecasts, 1)

mae_metrics = []
consumption_mae = []
production_mae = []
for item_id, ts in enumerate(test_dataset):
    ground_truth = np.array(ts["target"][-prediction_length:])
    mae = np.abs(forecast_median[item_id] - ground_truth).mean()
    if ts["feat_static_cat"][-1] > 0:
        consumption_mae.append(mae)
    else:
        production_mae.append(mae)
print(f"Consumption MAE: {np.mean(consumption_mae)}")
print(f"Production  MAE: {np.mean(production_mae)}")
print(f"Overall     MAE: {np.mean(production_mae + consumption_mae)}")

def plot(ts_index):
    fig, ax = plt.subplots()

    index = pd.period_range(
        start=test_dataset[ts_index][FieldName.START],
        periods=len(test_dataset[ts_index][FieldName.TARGET]),
        freq=freq,
    ).to_timestamp()

    # Major ticks every half year, minor ticks every month,
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())

    ax.plot(
        index[-2 * prediction_length :],
        test_dataset[ts_index]["target"][-2 * prediction_length :],
        label="actual",
    )

    plt.plot(
        index[-prediction_length:],
        np.median(forecasts[ts_index], axis=0),
        label="median",
    )
    plt.fill_between(
        index[-prediction_length:],
        forecasts[ts_index].mean(0) - forecasts[ts_index].std(axis=0),
        forecasts[ts_index].mean(0) + forecasts[ts_index].std(axis=0),
        alpha=0.3,
        interpolate=True,
        label="+/- 1-std",
    )

    plt.legend()
    plt.show()


#
# for ts_index, ts in enumerate(forecasts):
#     plot(ts_index)
