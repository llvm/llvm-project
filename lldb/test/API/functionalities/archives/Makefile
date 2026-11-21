C_SOURCES := main.c
MAKE_DSYM := NO
LD_EXTRAS := libfoo.a

all: a.out libbar.a libfoo-thin.a

include Makefile.rules

a.out: libfoo.a

# The archive members are not in C_SOURCES so that they are linked through
# libfoo.a instead of directly into a.out.
a.o b.o c.o: %.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

libfoo.a: a.o b.o
	$(AR) $(ARFLAGS) $@ $^

# This tests whether lldb can load a thin archive
libbar.a: c.o
	$(eval LLVM_ARFLAGS := -rcsDT)
	$(LLVM_AR) $(LLVM_ARFLAGS) $@ $^

libfoo-thin.a: a.o b.o
	$(eval LLVM_ARFLAGS := -rcsUT)
	$(LLVM_AR) $(LLVM_ARFLAGS) $@ $^
