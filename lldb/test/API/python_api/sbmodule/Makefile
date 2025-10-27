C_SOURCES := main.c a.c b.c
MAKE_DSYM := NO

all: a.out 

a.out: main.o libfoo.a
	$(LD) $(LDFLAGS) $^ -o $@

libfoo.a: a.o b.o
	$(AR) $(ARFLAGS) $@ $^

include Makefile.rules
