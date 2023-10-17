# This Makefile recursively calls itself, hence the ?=.
SWIFT_SOURCES ?= main.swift
SWIFTFLAGS_EXTRAS ?= -target $(TRIPLE) -I$(BUILDDIR)
all: Dylib $(EXE)

include Makefile.rules

.PHONY: Dylib
Dylib:
	$(MAKE) MAKE_DSYM=$(MAKE_DSYM) CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		VPATH=$(SRCDIR) -I $(SRCDIR) \
		SWIFT_SOURCES= \
		SWIFTFLAGS_EXTRAS="-target $(DYLIB_TRIPLE) -I__PATH_FROM_DYLIB__" \
		-f $(SRCDIR)/Makefile \
		DYLIB_FILENAME=Dylib.dylib \
		DYLIB_SWIFT_SOURCES=Dylib.swift \
		DYLIB_NAME=Dylib \
		DYLIB_ONLY=YES \
		LD_EXTRAS="-lSwiftCore" \
		Dylib.dylib

