CXX_SOURCES := main-copy.cpp
LD_EXTRAS := -Wl,-rpath "-Wl,$(shell pwd)"
USE_LIBDL :=1

a.out: libother

include Makefile.rules

# We copy the source files to move them to test source mapping
other-copy.c: other.c
	cp -f $< $@

main-copy.cpp: main.cpp
	cp -f $< $@

# The following shared library will be used to test breakpoints under dynamic loading
libother:  other-copy.c
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES DYLIB_C_SOURCES=other-copy.c DYLIB_NAME=other 
