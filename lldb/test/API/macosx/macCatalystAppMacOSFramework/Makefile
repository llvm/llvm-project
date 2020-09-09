C_SOURCES := main.c
LD_EXTRAS := -L. -lfoo

CFLAGS_EXTRAS := -target $(ARCH)-apple-ios13.0-macabi

# FIXME: rdar://problem/54986190
override CC=xcrun clang

all: libfoo.dylib a.out

libfoo.dylib: foo.c
	$(MAKE) -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES DYLIB_NAME=foo DYLIB_C_SOURCES=foo.c

include Makefile.rules
