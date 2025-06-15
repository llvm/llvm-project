CXX_SOURCES := main.cpp

all: a.out jit.out

include Makefile.rules


jit.out:
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		CXX_SOURCES=jit.cpp EXE=jit.out
