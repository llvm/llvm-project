C_SOURCES := main.c a.c b.c
EXE :=  # Define a.out explicitly
MAKE_DSYM := NO

all: a.out

a.out: main.o libfoo.a
	$(LD) $(LDFLAGS) $^ -o $@

libfoo.a: a.o b.o
	$(AR) $(ARFLAGS) $@ $^
	$(RM) $^

include Makefile.rules
