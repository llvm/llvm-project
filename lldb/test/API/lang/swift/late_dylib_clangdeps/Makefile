# This Makefile recursively calls itself, hence the ?=.
EXE ?= a.out
C_SOURCES ?= loader.c
all: dylib $(EXE)

include Makefile.rules

.PHONY: dylib
dylib:
	"$(MAKE)" MAKE_DSYM=$(MAKE_DSYM) CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		VPATH=$(SRCDIR) -I $(SRCDIR) \
		-f $(SRCDIR)/Makefile \
		DYLIB_FILENAME=dylib.dylib \
		DYLIB_SWIFT_SOURCES=dylib.swift \
		DYLIB_NAME=dylib \
		DYLIB_ONLY=YES \
		SWIFTFLAGS_EXTRAS="-Xcc -I$(SRCDIR)" \
		C_SOURCES= \
		LD_EXTRAS="-lSwiftCore -Xlinker -exported_symbol -Xlinker _f" \
		dylib.dylib

