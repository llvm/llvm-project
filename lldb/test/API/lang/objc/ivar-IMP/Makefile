CFLAGS := -g -O0
CFLAGS_NO_DEBUG = 

all: aout

aout: 
	$(CC) $(CFLAGS_NO_DEBUG) $(SRCDIR)/myclass.m -c -o myclass.o
	$(CC) $(CFLAGS) myclass.o $(SRCDIR)/repro.m -framework Foundation

clean::
	rm -f myclass.o

include Makefile.rules
