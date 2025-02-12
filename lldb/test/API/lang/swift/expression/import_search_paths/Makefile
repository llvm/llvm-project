SWIFT_SOURCES := main.swift
SWIFTFLAGS_EXTRAS = -I$(BUILDDIR)
LD_EXTRAS = -L$(BUILDDIR) -lDirect

all: libDirect $(EXE)
.phony: libDirect libHidden

include Makefile.rules

libHidden: $(SRCDIR)/hidden/Indirect.swift
	mkdir -p $(BUILDDIR)/hidden
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		-C $(BUILDDIR)/hidden VPATH=$(SRCDIR)/hidden \
		DYLIB_ONLY=YES \
		DYLIB_NAME=Indirect \
		DYLIB_SWIFT_SOURCES=Indirect.swift \
		DYLIB_MODULENAME=Indirect \
		SWIFTFLAGS_EXTRAS="-gnone" \
		MAKE_DSYM=NO

libDirect: $(SRCDIR)/Direct.swift libHidden
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES \
		DYLIB_NAME=Direct \
		DYLIB_SWIFT_SOURCES=Direct.swift \
		DYLIB_MODULENAME=Direct \
		SWIFTFLAGS_EXTRAS="-I$(BUILDDIR)/hidden" \
		LD_EXTRAS="-L$(BUILDDIR)/hidden -lIndirect" \
		MAKE_DSYM=NO

