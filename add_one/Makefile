.PHONY: build tag push all
#本地编译
nvcc:
	nvcc -o cuda_add_one cuda_add_one.cu

#本地运行
run:
	./cuda_add_one

build:
	sudo docker build -t cuda_add_one .
tag:
	sudo docker tag cuda_add_one 192.168.70.202:32373/registry/cuda_add_one
push:
	sudo docker push 192.168.70.202:32373/registry/cuda_add_one
all: build tag push

