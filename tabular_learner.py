from fastai.tabular.all import *
from pyxtend import struct
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

path = untar_data(URLs.ADULT_SAMPLE)

df = pd.read_csv(path/'adult.csv')

train_df, test_df = train_test_split(df, random_state=42)

dep_var = 'salary'

continuous_vars, categorical_vars = cont_cat_split(train_df, dep_var=dep_var)

categorical_vars.remove('native-country')

preprocessing = [Categorify, FillMissing, Normalize]


def no_split(obj):
    """
    Put everything in the train set
    """
    return list(range(len(obj))), []


splits = no_split(range_of(train_df))

full_df = pd.concat([train_df, test_df])

val_indices = list(range(len(train_df),len(train_df) + len(test_df)))

ind_splitter = IndexSplitter(val_indices)

splits = ind_splitter(full_df)

df_wrapper = TabularPandas(full_df, procs=preprocessing, cat_names=categorical_vars, cont_names=continuous_vars,
                   y_names=dep_var, splits=splits)

X_train, y_train = df_wrapper.train.xs, df_wrapper.train.ys.values.ravel()
X_test, y_test = df_wrapper.valid.xs, df_wrapper.valid.ys.values.ravel()

batch_size = 128
dls = df_wrapper.dataloaders(bs=batch_size)


batch = next(iter(dls.train))
cat_vars, cont_vars, labels = batch


learn = tabular_learner(dls, layers=[20,10])


learn.fit(1, 1e-2)

print(learn.summary())

