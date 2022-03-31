CXX_SOURCES := main.cpp
ENABLE_THREADS := YES
LD_EXTRAS := -L. -lbaz -I.

override ARCH := $(shell uname -m)

all: libbaz.dylib a.out

libbaz.dylib: baz.c
	$(MAKE) -f $(MAKEFILE_RULES) ARCH=$(ARCH) \
		DYLIB_ONLY=YES DYLIB_NAME=baz DYLIB_C_SOURCES=baz.c

include Makefile.rules
