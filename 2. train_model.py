import numpy as np
import os
from models import mitchnet
from sklearn.model_selection import train_test_split

FILE_I_END = 1860

WIDTH = 480
HEIGHT = 270
LR = 1e-3
EPOCHS = 30

MODEL_NAME = 'MITCH'

LOAD_MODEL = False
wl = 0
sl = 0
al = 0
dl = 0

wal = 0
wdl = 0
sal = 0
sdl = 0
nkl = 0

w = [1, 0, 0, 0, 0, 0, 0, 0, 0]
s = [0, 1, 0, 0, 0, 0, 0, 0, 0]
a = [0, 0, 1, 0, 0, 0, 0, 0, 0]
d = [0, 0, 0, 1, 0, 0, 0, 0, 0]
wa = [0, 0, 0, 0, 1, 0, 0, 0, 0]
wd = [0, 0, 0, 0, 0, 1, 0, 0, 0]
sa = [0, 0, 0, 0, 0, 0, 1, 0, 0]
sd = [0, 0, 0, 0, 0, 0, 0, 1, 0]
nk = [0, 0, 0, 0, 0, 0, 0, 0, 1]

# model = googlenet(WIDTH, HEIGHT, 3, LR, output=9, model_name=MODEL_NAME)
model = mitchnet(WIDTH, HEIGHT, 3, LR, output=9, model_name=MODEL_NAME)

training_data_read = [np.load(file_name, allow_pickle=True) for file_name in os.listdir() if
                      file_name.endswith('.npy')]
training_data_read_flat = [np.array(item) for sublist in training_data_read for item in sublist]

# iterates through the training files
for i in range(EPOCHS):
    print(f"EPOCH {i} OF {EPOCHS}")
    try:
        X = np.array([i[0] for i in training_data_read_flat]).reshape(-1, WIDTH, HEIGHT, 3)
        y = [i[1] for i in training_data_read_flat]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model.fit(
            {'input': X},
            {'targets': y},
            n_epoch=1,
            validation_set=({'input': X_test}, {'targets': y_test}),
            snapshot_step=2500,
            show_metric=True,
            run_id=MODEL_NAME
        )

        if i % 10 == 0:
            print('SAVING MODEL!')
            model.save(MODEL_NAME)

    except Exception as e:
        print(str(e))

    print(f"FINISHED EPOCH {i} OF {EPOCHS}")
    print()

# tensorboard --logdir=foo:J:/phase10-code/log
