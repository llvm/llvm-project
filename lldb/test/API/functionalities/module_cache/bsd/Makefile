C_SOURCES := main.c a.c b.c c.c
EXE :=  # Define a.out explicitly
MAKE_DSYM := NO

all: a.out

a.out: main.o libfoo.a
	$(LD) $(LDFLAGS) $^ -o $@

lib_ab.a: a.o b.o
	$(AR) $(ARFLAGS) $@ $^
	$(RM) $^

# Here we make a .a file that has two a.o files with different modification
# times and different content by first creating libfoo.a with only a.o and b.o,
# then we sleep for 2 seconds, touch c.o to ensure it has a different
# modification time, and then rename c.o to a.o and then add it to the .a file
# again. This is to help test that the module cache will create different
# directories for the two different a.o files.
libfoo.a: lib_ab.a c.o
	sleep 2
	touch c.o
	mv c.o a.o
	$(AR) $(ARFLAGS) $@ lib_ab.a a.o
	$(RM) a.o

include Makefile.rules
