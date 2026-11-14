C_SOURCES := main.c
MAKE_DSYM := NO
LD_EXTRAS := libfoo.a

include Makefile.rules

a.out: libfoo.a

# The archive members are not in C_SOURCES so that they are linked through
# libfoo.a instead of directly into a.out.
a.o b.o: %.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

libfoo.a: a.o b.o
	$(AR) $(ARFLAGS) $@ $^
