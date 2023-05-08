import dill
import pandas as pd
import json
import os
from datetime import datetime
from pathlib import Path
import logging

path = os.environ.get('PROJECT_PATH', '.')


def predict():

    current_pkl = list(Path(f'{path}/data/models/').glob('*.pkl'))[-1]
    with open(current_pkl, 'rb') as file:
        model = dill.load(file)
    df_pred = pd.DataFrame(columns=['car_id', 'pred'])
    for jsonfile in Path(f'{path}/data/test/').glob('*.json'):
        with open(jsonfile) as json_file:
            jsn = json.load(json_file)
            df = pd.DataFrame.from_dict([jsn])
            y = model.predict(df)
            df_tmp = pd.DataFrame({'car_id': df.id, 'pred': y})
            df_pred = pd.concat([df_pred, df_tmp], axis=0)
    df_pred.to_csv(f'{path}/data/predictions/pred_{datetime.now().strftime("%Y%m%d%H%M")}.csv')
    logging.info(f'CSV is saved to {path}/data/predictions/')


if __name__ == '__main__':
    predict()
