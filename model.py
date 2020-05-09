import numpy as np


def reshape(data, size=7):
	d = np.array(
		[data[tup[0]:tup[0] + size] for tup in enumerate(data) if (tup[0] + size) <= len(data)])
	return d


def transform_diff(data):
	# transforms data into the relative differences
	def diff(ind):
		curr = data[ind][0]
		prev = data[ind - 1][0] if ind - 1 >= 0 else curr
		diff = curr - prev
		return diff

	d = np.array([[diff(tup[0])] for tup in enumerate(data)])
	return d


def transform_perc(data):
	# transforms data into the percentage increase
	def perc(ind):
		curr = data[ind][0]
		prev = data[ind - 1][0] if ind - 1 >= 0 else 0
		perc = ((curr/(prev/100))/100)-1
		perc = perc if ind - 1 >= 0 else 1
		# print(f"i={ind}; curr={curr}; prev={prev}; perc = {perc}")
		return perc

	d = np.array([[perc(tup[0])] for tup in enumerate(data)])
	# print(f"Out >> {d}")
	return d


def train(data, test_size=1, obv_size=7, tr_size=3):
	test_size = test_size + 1
	shaped_data = reshape(data, tr_size)
	# transformed_y = transform_perc([[arr[-1]] for arr in data[tr_size:]])
	transformed_y = [[arr[-1]] for arr in data[tr_size:]]

	x_train = np.array(shaped_data[:-test_size])
	y_train = np.array(transformed_y[: -(test_size - 1)])

	x_test = np.array(shaped_data[-test_size:-1])
	y_test = np.array(transformed_y[-(test_size - 1)])

	x_sample = np.array([shaped_data[-1]])

	print(f"data {data.shape} : {data}]")
	print(shaped_data.shape)
	print("x_train :", x_train.shape, " : ", x_train)
	print("y_train :", y_train.shape, " : ", y_train)

	from keras.models import Sequential
	from keras.layers import Dense, LSTM, Dropout
	from keras.callbacks import EarlyStopping
	from keras.utils import plot_model
	from keras.optimizers import adam

	model = Sequential()
	model.add(LSTM(units=7, activation="relu",
	               input_shape=(tr_size, obv_size)
	               ))
	model.add(Dense(activation="relu", units=4))
	model.add(Dense(activation="linear", units=1))

	model.compile(metrics=['accuracy', 'mae'], optimizer='adadelta',
	              loss='mae')  # adadelta > adamax > adam > nadam > sgd > adagrad
	print(model.summary())

	callbacks = [EarlyStopping(
		monitor="val_mean_absolute_error", patience=500, min_delta=0.0000000001,
		verbose=1  # , restore_best_weights=True
		)]
	fitted = model.fit(
		x_train, y_train,
		epochs=10000, batch_size=8,
		validation_split=0.1, verbose=2,
		callbacks=callbacks
		)
	# print("Fitted : {}".format(fitted.history))

	plot_model(
		model, to_file="model.png",
		show_layer_names=True, show_shapes=True
		)
	model.save("model.h5")

	print("Testing...")
	print(f"Inp : {x_test}")
	print(f"pred : {model.predict(x_test)}; >> true : {y_test}")
	print(f"Result (loss, acc, mae): {model.evaluate(x_test, y_test)}")

	print("\nPredicting...")
	pred = model.predict(x_sample)
	last = x_sample[0][tr_size-1][obv_size-1]
	forward = last + (last * pred[0][0])
	print(f">> Inp : {x_sample};\n>> Out : {pred}")
	print(f"Prediction Normalized : {last}+({last}*{pred[0][0]}) => {forward} new cases);")


if __name__ == "__main__":
	# is not imported
	import get_data

	obv_size = 2
	tr_size = 5
	data = get_data.get(obv_size, use_increase=True)
	train(data, 1, obv_size, tr_size)
