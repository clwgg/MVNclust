OUT= mvnclust

shared: main.c
	gcc -O2 -Wall $^ -lgsl -lgslcblas -lm -o $(OUT)

static: main.c ./gsl/.libs/libgsl.a ./gsl/cblas/.libs/libgslcblas.a
	gcc -O2 -Wall $^ -I./gsl/ -lm -o $(OUT)

submodules:
	cd gsl; ./autogen.sh; ./configure; make; make libgsl.la; cd ..

delsubmod:
	git submodule deinit -f -- gsl
	rm -rf .git/modules/gsl
	git rm -f gsl
