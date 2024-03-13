# Makefile for building the SVM-HMM model files

# Target to download svm_hmm.tar.gz
svm_hmm_package_download:
	curl -O https://osmot.cs.cornell.edu/svm_hmm/current/svm_hmm.tar.gz
	mkdir -p svm_hmm 
	tar -xvzf svm_hmm.tar.gz -C svm_hmm

build:
	cd ./svm_hmm && make 'CC = /usr/bin/gcc'

# Target to clean up downloaded files
clean:
	rm -rf svm_hmm.tar.gz svm_hmm Results __pycache__ .DS_Store modelfile.dat svm_mc_data