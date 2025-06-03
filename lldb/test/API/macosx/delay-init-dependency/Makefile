C_SOURCES := main.c
LD_EXTRAS := -L. -Wl,-delay_library,libfoo.dylib

.PHONY: build-libfoo
all: build-libfoo a.out

include Makefile.rules

build-libfoo: foo.c
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_C_SOURCES=foo.c DYLIB_NAME=foo DYLIB_ONLY=YES
