.PHONY: setup run clean

setup:
	python -m pip install -r requirements.txt
	mkdir -p uploads
	mkdir -p static/images
	# Unzip the COCO images if they exist
	test -f coco_images_resized.zip && unzip -o coco_images_resized.zip -d static/images || echo "No image zip found"

run:
	python app.py

clean:
	rm -rf uploads/*
	rm -rf static/images/*
	rm -rf __pycache__