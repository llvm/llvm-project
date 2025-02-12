SWIFT_SOURCES := main.swift
SWIFT_CXX_INTEROP := 1
SWIFTFLAGS_EXTRAS = -I.
LD_EXTRAS = -L$(BUILDDIR) -lDylib

all: libDylib.dylib a.out

include Makefile.rules

libDylib.dylib: Dylib.swift
	"$(MAKE)" MAKE_DSYM=$(MAKE_DSYM) CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		VPATH=$(SRCDIR) -I $(SRCDIR) \
		-f $(THIS_FILE_DIR)/Makefile.rules \
		DYLIB_SWIFT_SOURCES=Dylib.swift \
		DYLIB_NAME=Dylib \
		DYLIB_ONLY=YES \
		SWIFT_SOURCES= \
		SWIFT_CXX_INTEROP=0 \
		SWIFT_BRIDGING_HEADER= \
		all
