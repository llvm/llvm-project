SWIFT_SOURCES = main.swift
SWIFTFLAGS_EXTRAS = -I.
SWIFT_BRIDGING_HEADER = bridging-header.h
LD_EXTRAS = -L. -lDylib

all: dylib $(EXE)

include Makefile.rules

.PHONY: dylib
dylib:
	$(MAKE) MAKE_DSYM=$(MAKE_DSYM) CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		VPATH=$(SRCDIR) -I $(SRCDIR) \
		-f $(THIS_FILE_DIR)/Makefile.rules \
		DYLIB_C_SOURCES=dylib.c \
		DYLIB_NAME=Dylib \
		DYLIB_ONLY=YES \
		DEBUG_INFO_FLAG= \
		all
