
mvnclust: main.c ./gsl/.libs/libgsl.a ./gsl/cblas/.libs/libgslcblas.a
	gcc -O2 -Wall $^ -I./gsl/ -lm -o $@

submodules:
	cd gsl; ./autogen.sh; ./configure; make; make libgsl.la; cd ..

