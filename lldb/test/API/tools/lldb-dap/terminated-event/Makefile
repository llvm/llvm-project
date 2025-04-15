DYLIB_NAME := foo
DYLIB_CXX_SOURCES := foo.cpp
CXX_SOURCES := main.cpp

LD_EXTRAS := -Wl,-rpath "-Wl,$(shell pwd)"
USE_LIBDL :=1

include Makefile.rules

all: a.out.stripped

a.out.stripped:
	$(STRIP) -o a.out.stripped a.out

ifneq "$(CODESIGN)" ""
	$(CODESIGN) -fs - a.out.stripped
endif