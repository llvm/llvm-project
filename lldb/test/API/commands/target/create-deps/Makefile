LD_EXTRAS := -L. -lload_a
CXX_SOURCES := main.cpp

a.out: libload_a

include Makefile.rules

libload_a:
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES DYLIB_NAME=load_a DYLIB_CXX_SOURCES=a.cpp
