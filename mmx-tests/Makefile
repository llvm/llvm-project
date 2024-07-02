USE_XMM=
#USE_XMM=--use-xmm

OLDCC ?= clang-10
NEWCC ?= ../build/bin/clang
TESTCC=$(OLDCC)
COPTS ?=

gen_orig.c: mmx-tests.py
	./mmx-tests.py --kind=wrapper --wrapper-prefix=orig $(USE_XMM) > $@
gen_orig.h: mmx-tests.py
	./mmx-tests.py --kind=wrapper_h --wrapper-prefix=orig $(USE_XMM) > $@
gen_new.c: mmx-tests.py
	./mmx-tests.py --kind=wrapper --wrapper-prefix=new $(USE_XMM) > $@
gen_new.h: mmx-tests.py
	./mmx-tests.py --kind=wrapper_h --wrapper-prefix=new $(USE_XMM) > $@
gen_test.inc: mmx-tests.py
	./mmx-tests.py --kind=test $(USE_XMM) > $@
gen_orig.o: gen_orig.c
	$(OLDCC) -c $(COPTS) -O2 -o $@ $^
gen_new.o: gen_new.c
	$(NEWCC) -c $(COPTS) -O2 -o $@ $^
test.o: test.c gen_test.inc gen_orig.h gen_new.h
	$(TESTCC) -c $(COPTS) -o $@ test.c
test: test.o gen_orig.o gen_new.o
	$(TESTCC) $(COPTS) -o $@ $^ -lm

clean:
	rm -f gen_orig.c gen_orig.h gen_new.c gen_new.h gen_test.inc gen_orig.o gen_new.o test.o test
