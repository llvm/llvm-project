CXX_SOURCES := main.cpp

all: secondprog

include Makefile.rules

secondprog: secondprog.cpp
	$(MAKE) -f $(MAKEFILE_RULES) \
		CXX_SOURCES=secondprog.cpp EXE=secondprog
