C_SOURCES := main.c
LD_EXTRAS = hidden.o

include Makefile.rules

a.out: hidden.o

hidden.o: hidden.c
	$(CC) $(CFLAGS) -g0 -c -o $@ $<

