# This Makefile recursively calls itself, hence the ?=.
EXE ?= a.out
SWIFT_SOURCES ?= loader.swift
SWIFT_BRIDGING_HEADER ?= ClangHeader.h
SWIFT_PRECOMPILE_BRIDGING_HEADER ?= NO
SWIFTFLAGS_EXTRAS ?= -enable-bare-slash-regex

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
		SWIFT_SOURCES= \
		SWIFT_BRIDGING_HEADER= \
		LD_EXTRAS="-lSwiftCore -Xlinker -exported_symbol -Xlinker _f" \
		dylib.dylib

