C_SOURCES := main.c

# Make an archive that has two object files with the same name, but
# different timestamps. Do it all in one rule so that the timestamps
# can be controlled without confusing Make.
libfoo.a: a.c sub1/a.c
	$(CC) $(CFLAGS) -c $(<D)/a.c -o a.o
	mkdir -p sub1
	$(CC) $(CFLAGS) -c $(<D)/sub1/a.c -o sub1/a.o
	touch -t '198001010000.00' a.o
	touch -t '198001010000.01' sub1/a.o
	$(AR) $(ARFLAGS) $@ a.o sub1/a.o
	rm a.o sub1/a.o

include Makefile.rules

# Needs to come after include
OBJECTS += libfoo.a
$(EXE) : libfoo.a
.DEFAULT_GOAL := $(EXE)
