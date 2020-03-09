C_SOURCES := main.c
CFLAGS_EXTRAS := -std=c99 -fmodules
LD_EXTRAS := -ldylib -L.

all: a.out hidden/libdylib.dylib

a.out: libdylib.dylib

include Makefile.rules

libdylib.dylib: dylib.c
	$(MAKE) -C $(BUILDDIR) -f $(MAKEFILE_RULES) \
		C_SOURCES= DYLIB_C_SOURCES=dylib.c DYLIB_NAME=dylib \
		CFLAGS_EXTRAS=-DHAS_THEM LD_EXTRAS=-dynamiclib

hidden/libdylib.dylib:
	mkdir hidden
	$(MAKE) -C $(BUILDDIR)/hidden -f $(MAKEFILE_RULES) \
		C_SOURCES= DYLIB_C_SOURCES=dylib.c DYLIB_NAME=dylib \
		LD_EXTRAS=-dynamiclib
