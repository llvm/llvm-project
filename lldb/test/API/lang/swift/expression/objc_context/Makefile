OBJC_SOURCES := main.m
CFLAGS_EXTRAS = $(MANDATORY_MODULE_BUILD_CFLAGS) -I$(BUILDDIR)
LD_EXTRAS = -lFoo -L$(BUILDDIR)

# This test builds an Objective-C main program that imports two Swift
# .dylibs with conflicting ClangImporter search paths.

all: libFoo.dylib $(EXE)

include Makefile.rules

lib%.dylib: %.swift
	"$(MAKE)" MAKE_DSYM=YES CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		BASENAME=$(shell basename $< .swift) \
		VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/dylib.mk all

clean::
	rm -rf *.swiftmodule *.swiftdoc *.dSYM *~ lib*.dylib a.out *.o
