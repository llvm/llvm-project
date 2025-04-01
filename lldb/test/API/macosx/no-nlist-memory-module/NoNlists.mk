DYLIB_ONLY := YES
DYLIB_NAME := no-nlists
DYLIB_C_SOURCES := no-nlists.c
DYLIB_OBJECTS += no-nlist-sect.o

no-nlist-sect.o:
	$(CC) $(CFLAGS) -c -o no-nlist-sect.o $(SRCDIR)/no-nlist-sect.s

include Makefile.rules

clean::
	rm -rf *.o *.dylib a.out *.dSYM
