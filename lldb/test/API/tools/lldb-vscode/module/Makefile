DYLIB_NAME := foo
DYLIB_CXX_SOURCES := foo.cpp
CXX_SOURCES := main.cpp

all: a.out.stripped

include Makefile.rules

a.out.stripped: a.out.dSYM
	strip -o a.out.stripped a.out
ifneq "$(CODESIGN)" ""
	$(CODESIGN) -fs - a.out.stripped
endif
