SWIFT_SOURCES := main.swift
SWIFT_ENABLE_EXPLICIT_MODULES := YES
SWIFTFLAGS_EXTRAS := -I.
LD_EXTRAS = -L$(BUILDDIR) -lDylib

all: libDylib a.out

include Makefile.rules

# Note that this is being built with implicit modules!
libDylib: Dylib.swift
	mkdir -p $(BUILDDIR)/$(shell basename $< .swift)
	"$(MAKE)" MAKE_DSYM=YES CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		DYLIB_NAME=Dylib DYLIB_SWIFT_SOURCES=Dylib.swift \
		VPATH=$(SRCDIR) SWIFTFLAGS_EXTRAS=-I$(SRCDIR) \
		-f $(MAKEFILE_RULES) all
