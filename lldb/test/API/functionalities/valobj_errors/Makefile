C_SOURCES := main.c
LD_EXTRAS = hidden.o

a.out: hidden.o

hidden.o: hidden.c
	$(CC) -g0 -c -o $@ $<

include Makefile.rules
