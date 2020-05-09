import json
import numpy as np


def get(size, country_backup="", use_increase=False):
	with open("data.json", "r+") as file:
		data = json.load(file)
		raw = data['data'] if not country_backup else data['backups'][country_backup]
		vals = [raw[val][1 if use_increase else 0] for val in raw]
		# vals = full data array
		chunked = [vals[key[0]:key[0]+size] for key in enumerate(vals) if key[0]+size <= len(vals)]
		# chunked = grouped nested array of data values with each array having length = size
		return np.array(chunked)


def refill(country, start_date="01-01", use_saved=False, backup_only=True):
	import requests
	from bs4 import BeautifulSoup
	import json
	import re

	start_date = int("".join(start_date.split("-")))
	if not use_saved:
		url = f"https://api.covid19api.com/dayone/country/{country}/status/confirmed"
		html = requests.get(url)
		html.raise_for_status()  # This will raise an error, if a problem occured while fetching the
		# webpage
		content = html.content
		soup = BeautifulSoup(content, 'html.parser')
		text = soup.string
	else:
		# read from saved copy instead of calling api again
		print("Using saved API response...")
		with open("api.txt", "r") as file:
			text = file.read()

	# save a copy of api response
	with open("api.txt", "w+") as file:
		file.write(text)
	res = json.loads(text)
	data = {}
	print(res)
	response_country = ""
	for k, obj in enumerate(res):
		if obj["Country"]:
			key = obj["Date"]
			val = obj["Cases"]
			inc = (int(val) - int(0 if k == 0 else int(res[k - 1]["Cases"])))
			inc = inc if inc != 0 else 1
			date = re.findall("-\d+-\d+", key)[0][1:]
			if int("".join(date.split("-"))) >= start_date:
				data[date] = [val, inc]
				response_country = obj["Country"]
	print(data)  #

	with open("data.json", "r") as file:
		old = json.loads(file.read())
		old["backups"][response_country] = data
		if not backup_only:
			old["data"] = data
		with open("data.json", "w") as file2:
			file2.write(json.dumps(old))


if __name__ == "__main__":
	# United States of America; Italy; Nigeria
	# 03-16; 01-01
	# refill("Nigeria", start_date="03-16", use_saved=False, backup_only=False)
	print(get(3, "Nigeria"))
