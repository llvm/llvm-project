SWIFT_SOURCES := main.swift
SWIFTFLAGS_EXTRAS = -I$(BUILDDIR)
LD_EXTRAS = -L$(BUILDDIR) -lGeneric -lBuilder

all: libBuilder.dylib $(EXE)

include Makefile.rules

libGeneric.dylib: $(SRCDIR)/Generic.swift
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES \
		DYLIB_NAME=Generic \
		DYLIB_SWIFT_SOURCES=Generic.swift \
		DYLIB_MODULENAME=Generic \
		SWIFTFLAGS_EXTRAS=-enable-library-evolution

libBuilder.dylib: $(SRCDIR)/Builder.swift libGeneric.dylib
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES \
		DYLIB_NAME=Builder \
		DYLIB_SWIFT_SOURCES=Builder.swift \
		DYLIB_MODULENAME=Builder \
		DYLIB_HIDE_SWIFTMODULE=YES \
		SWIFTFLAGS_EXTRAS="-I$(BUILDDIR)" \
		LD_EXTRAS="-L$(BUILDDIR) -lGeneric" \
		MAKE_DSYM=NO
	rm -f $(BUILDDIR)/Builder.swiftmodule
