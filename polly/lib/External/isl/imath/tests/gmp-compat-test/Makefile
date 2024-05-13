CFLAGS+=-fPIC -I$(IMATH_DIR)
IMATH_DIR=../..

runtest: imath_test.so gmp_test.so wrappers.py random.tests
	./runtest $(TESTS)

gmp_test.c: gmp_custom_test.c genctest.py gmpapi.py
	./genctest.py gmp > $@

imath_test.c: imath_custom_test.c genctest.py gmpapi.py
	./genctest.py imath > $@

gmp_test.so: gmp_test.o
	$(CC) $(CFLAGS) -shared -o $@ $^ -lgmp

imath_test.so: imath_test.o
	$(CC) $(CFLAGS) -shared -L$(IMATH_DIR) -o $@ $^ -limath

wrappers.py: genpytest.py gmpapi.py
	./genpytest.py > $@

random.tests: gendata.py
	./gendata.py > $@

clean:
	rm -f a.out *.so *.o gmp_test.c imath_test.c wrappers.py *.pyc
	rm -rf __pycache__
