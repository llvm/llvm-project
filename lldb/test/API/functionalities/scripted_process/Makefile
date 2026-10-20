CXX_SOURCES := main.cpp
ENABLE_THREADS := YES
LD_EXTRAS := -L. -lbaz -I.

override TRIPLE := $(shell $(CC) -dumpmachine)

CXXFLAGS_EXTRAS := -target $(TRIPLE)

all: libbaz.dylib a.out

libbaz.dylib: baz.cpp
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES DYLIB_NAME=baz DYLIB_CXX_SOURCES=baz.cpp

include Makefile.rules
