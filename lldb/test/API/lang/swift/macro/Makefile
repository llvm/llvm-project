SWIFT_SOURCES := main.swift
SWIFTFLAGS_EXTRAS = -enable-experimental-feature Macros \
	            -load-plugin-library $(BUILDDIR)/libMacroImpl.dylib

all: libMacroImpl.dylib $(EXE)

include Makefile.rules

libMacroImpl.dylib:
	$(MAKE) MAKE_DSYM=$(MAKE_DSYM) CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		VPATH=$(SRCDIR) -I $(SRCDIR) \
		-f $(THIS_FILE_DIR)/Makefile.rules \
		DYLIB_SWIFT_SOURCES=MacroImpl.swift \
		DYLIB_NAME=MacroImpl \
		DYLIB_ONLY=YES \
		SWIFTFLAGS_EXTRAS="-I$(SWIFT_HOSTDIR)" \
		LD_EXTRAS="-L$(SWIFT_HOSTDIR)" \
		SWIFT_SOURCES= \
		all

