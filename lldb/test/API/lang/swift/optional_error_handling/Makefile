SWIFT_SOURCES := main.swift
SWIFTFLAGS_EXTRAS = -I.
LD_EXTRAS = -L. -lLibrary


all: Library $(EXE)

include Makefile.rules

.PHONY: Library
Library:
	$(MAKE) MAKE_DSYM=NO CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		VPATH=$(SRCDIR) -I $(SRCDIR) SRCDIR=$(SRCDIR) \
		-f $(THIS_FILE_DIR)/Makefile.rules \
		DYLIB_SWIFT_SOURCES=Library.swift \
		SWIFT_BRIDGING_HEADER=bridging.h \
		SWIFT_PRECOMPILE_BRIDGING_HEADER=NO \
		DYLIB_NAME=Library \
		DYLIB_ONLY=YES \
		DEBUG_INFO_FLAG= \
		all
