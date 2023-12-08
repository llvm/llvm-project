CXX_SOURCES := main.cpp
USE_LIBDL := 1

all: hidden_lib a.out

include Makefile.rules

hidden_lib:
	$(MAKE) VPATH=$(SRCDIR)/hidden -C hidden -f $(MAKEFILE_RULES) \
	DYLIB_ONLY=YES DYLIB_CXX_SOURCES=d.cpp DYLIB_NAME=loadunload

