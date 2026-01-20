CXX_SOURCES := main.cpp

.PHONY: build-libfoo
all: build-libfoo a.out

include Makefile.rules

build-libfoo: foo.c
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_C_SOURCES=foo.c DYLIB_NAME=foo DYLIB_ONLY=YES

