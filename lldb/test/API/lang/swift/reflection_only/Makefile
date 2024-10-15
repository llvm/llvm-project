SWIFT_SOURCES := main.swift
HIDE_SWIFTMODULE := YES

SWIFTFLAGS_EXTRAS := -I. -enable-library-evolution

LD_EXTRAS = -L. -Xlinker -rpath -Xlinker $(BUILDDIR) -ldynamic_lib

all: dylib $(EXE)
.PHONY: dylib

include Makefile.rules
dylib: lib.swift
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		MAKE_DSYM=YES DYLIB_ONLY=YES DYLIB_NAME=dynamic_lib \
		DYLIB_SWIFT_SOURCES="lib.swift" \
		DYLIB_HIDE_SWIFTMODULE=YES \
		SWIFTFLAGS_EXTRAS="-I$(BUILDDIR) $(SWIFTFLAGS_EXTRAS)"
