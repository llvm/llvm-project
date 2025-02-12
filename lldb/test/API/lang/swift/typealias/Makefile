SWIFT_SOURCES = main.swift
SWIFTFLAGS_EXTRAS = -I.
LD_EXTRAS = -L. -lDylib

all: dylib $(EXE)

include Makefile.rules

.PHONY: dylib
dylib:
	"$(MAKE)" MAKE_DSYM=$(MAKE_DSYM) CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		VPATH=$(SRCDIR) -I $(SRCDIR) \
		-f $(THIS_FILE_DIR)/Makefile.rules \
		DYLIB_SWIFT_SOURCES=Dylib.swift \
		DYLIB_NAME=Dylib \
		DYLIB_ONLY=YES \
		SWIFT_SOURCES= \
		LD_EXTRAS= \
		all

