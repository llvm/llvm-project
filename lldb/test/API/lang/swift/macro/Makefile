SWIFTFLAGS_EXTRAS = -I.

ifneq "$(SWIFT_SOURCES)" "empty.swift"

SWIFTFLAGS_EXTRAS += -load-plugin-library $(BUILDDIR)/libMacroImpl.dylib -dwarf-version=5
LD_EXTRAS = -L$(BUILDDIR) -lMacro

endif

all: libMacro.dylib libMacroImpl.dylib $(EXE)

include Makefile.rules

libMacro.dylib:
	"$(MAKE)" MAKE_DSYM=$(MAKE_DSYM) CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		VPATH=$(SRCDIR) -I $(SRCDIR) \
		-f $(THIS_FILE_DIR)/Makefile.rules \
		DYLIB_SWIFT_SOURCES=Macro.swift \
		DYLIB_NAME=Macro \
		DYLIB_ONLY=YES \
		SWIFT_SOURCES= \
		SWIFTFLAGS_EXTRAS="-load-plugin-library $(BUILDDIR)/libMacroImpl.dylib" \
		all
	$(RM) $(BUILDDIR)/Macro.swiftinterface

libMacroImpl.dylib:
	"$(MAKE)" MAKE_DSYM=$(MAKE_DSYM) CC=$(CC) SWIFTC=$(SWIFTC) \
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

