SWIFT_SOURCES := main.swift

SWIFTFLAGS_EXTRAS := -I.

LD_EXTRAS = -L. -Xlinker -rpath -Xlinker $(BUILDDIR) -lOpaqueLib

all: dylib $(EXE)
.PHONY: dylib

include Makefile.rules

dylib: lib.swift
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		MAKE_DSYM=YES DYLIB_ONLY=YES DYLIB_NAME=OpaqueLib \
		DYLIB_SWIFT_SOURCES="lib.swift" \
		SWIFTFLAGS_EXTRAS="-enable-library-evolution"
