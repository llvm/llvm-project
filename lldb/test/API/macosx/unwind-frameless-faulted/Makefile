C_SOURCES := main.c

interrupt-and-trap-funcs.o: interrupt-and-trap-funcs.c
	$(CC) $(CFLAGS) -E -o interrupt-and-trap-funcs.s $(SRCDIR)/interrupt-and-trap-funcs.c
	$(CC) $(CFLAGS) -c -o interrupt-and-trap-funcs.o interrupt-and-trap-funcs.s

include Makefile.rules

a.out: interrupt-and-trap-funcs.o

# Needs to come after include
OBJECTS += interrupt-and-trap-funcs.o

