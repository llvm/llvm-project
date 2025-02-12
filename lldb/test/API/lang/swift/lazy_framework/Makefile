SWIFT_SOURCES := main.swift

SWIFTFLAGS_EXTRAS = -F $(BUILDDIR)

all: Lazy.framework $(EXE)

include Makefile.rules

Lazy.framework: $(SRCDIR)/Lazy.swift
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_NAME=Lazy \
		DYLIB_SWIFT_SOURCES=Lazy.swift \
		DYLIB_MODULENAME=Lazy \
		FRAMEWORK=Lazy
	rm -f $(BUILDDIR)/Lazy.swiftmodule $(BUILDDIR)/Lazy.swiftinterface
