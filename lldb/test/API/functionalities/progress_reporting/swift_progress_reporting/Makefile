SWIFT_SOURCES := main.swift
SWIFT_BRIDGING_HEADER := bridging.h
SWIFT_PRECOMPILE_BRIDGING_HEADER := YES
LD_EXTRAS := -L. -linvisible
SWIFTFLAGS_EXTRAS := -I.

all: libinvisible.dylib $(EXE)

include Makefile.rules

libinvisible.dylib: Invisible.swift
	$(MAKE) -f $(MAKEFILE_RULES) \
		MAKE_DSYM=YES DYLIB_ONLY=YES DYLIB_NAME=Invisible \
		DYLIB_SWIFT_SOURCES="Invisible.swift" \
		SWIFTFLAGS_EXTRAS=-I$(BUILDDIR)
