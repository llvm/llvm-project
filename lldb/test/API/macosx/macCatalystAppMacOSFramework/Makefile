C_SOURCES := main.c
LD_EXTRAS := -L. -lfoo

override TRIPLE := $(ARCH)-apple-ios13.0-macabi
CFLAGS_EXTRAS := -target $(TRIPLE)

# FIXME: rdar://problem/54986190
override CC_TYPE=clang
override CC=xcrun clang

all: libfoo.dylib a.out

libfoo.dylib: foo.c
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES DYLIB_NAME=foo DYLIB_C_SOURCES=foo.c

include Makefile.rules
