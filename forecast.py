def predict(days):
	from get_data import get
	from keras.models import load_model
	from model import reshape
	import numpy as np

	obv_size = 2
	tr_size = 5
	raw_unshaped_total = get(obv_size, use_increase=False)
	raw_unshaped = get(obv_size, use_increase=True)
	raw = reshape(raw_unshaped, tr_size)
	x_test = raw[:-1]
	y_test = np.array([[arr[-1]] for arr in raw_unshaped[tr_size:]])
	model = load_model("model.h5")

	# print(f"{x_test} : {x_test.shape}")
	# print(f"{y_test} : {y_test.shape}")
	test_result = model.evaluate(x_test, y_test)
	mae_loss = test_result[2]

	last_known_seq = np.array([raw[-1]])
	pred = model.predict(last_known_seq)
	print(f"Tomorrow (mid) : {pred[0][0]}")

	last_known_day = last_known_seq[-1][-1][-1]
	last_known_total = raw_unshaped_total[-1][-1]
	print(f"Today = total: {last_known_total}; day: {last_known_day}")
	obj = {
		"min": [], "mid" : [], "max": [],
		"variance": mae_loss,
		"start" : (last_known_total, last_known_day),
		"min_end" : 0, "mid_end": 0, "max_end": 0
		}

	for key in obj:
		if key not in ['min', 'mid', 'max']:
			continue
		last_seq = last_known_seq
		last_total = last_known_total
		# last_day = last_known_day
		count = 0
		while count < days:
			count += 1
			# print(f"Day {count} sequence : {last_seq}")
			pred = model.predict(np.array(last_seq))[0][0]
			loss = mae_loss if key == 'max' else 0
			loss = -mae_loss if key == 'min' else loss
			pred = pred + loss
			last_day = pred
			last_total += pred
			obj[key+'_end'] = last_total
			obj[key].append(pred)
			last_seq = [last_seq[0][1:]]
			last_seq = np.append(last_seq[0], [last_day, pred])
			last_seq = np.array(last_seq.reshape((1, 5, 2)))

			# print(f"Day {count} pred : {pred}")
	print(f"{obj} >> min : {len(obj['min'])}; mid : {len(obj['mid'])}; max : {len(obj['max'])}")


predict(3)
