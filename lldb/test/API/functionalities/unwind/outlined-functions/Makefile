C_SOURCES := main.c

outlined-functions.o: outlined-functions.s
	$(CPP) -o outlined-functions.s $(SRCDIR)/outlined-functions.s
	$(CC) $(CFLAGS) -c -o outlined-functions.o outlined-functions.s

include Makefile.rules

a.out: outlined-functions.o

# Needs to come after include
OBJECTS += outlined-functions.o

