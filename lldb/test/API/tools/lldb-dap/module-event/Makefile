CXX_SOURCES := main.cpp
LD_EXTRAS := -Wl,-rpath "-Wl,$(shell pwd)"
USE_LIBDL :=1

a.out: libother

include Makefile.rules

# The following shared library will be used to test breakpoints under dynamic loading
libother:  other.c
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES DYLIB_C_SOURCES=other.c DYLIB_NAME=other
