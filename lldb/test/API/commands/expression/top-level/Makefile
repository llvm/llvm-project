CXX_SOURCES := main.cpp test.cpp

all: dummy

include Makefile.rules

dummy: dummy.cpp
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		CXX_SOURCES=dummy.cpp EXE=dummy

