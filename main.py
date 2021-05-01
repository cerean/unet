# my changes are denoted by my initials, {AC}

from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
trainGen = trainGenerator(2, 'data/membrane/train', 'image', 'label', data_gen_args, save_to_dir = 'data/membrane/train/aug')

model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.weights.hdf5', monitor='loss', verbose=1, save_best_only=True, save_weights_only=True)

# updated to model.fit() from depreciated model.fit_generator()
model.fit(trainGenerator, steps_per_epoch=2000, epochs=1, callbacks=[model_checkpoint])

testGen = testGenerator("data/membrane/test")

# updated to model.predict() from depreciated model.predict_generator() {AC}
results = model.predict(testGen, 30, verbose=1)
saveResult("data/membrane/test/predict", results)