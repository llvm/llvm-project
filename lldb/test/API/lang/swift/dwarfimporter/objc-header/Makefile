
MAKE_GMODULES := YES
SWIFT_OBJC_INTEROP = 1
SWIFT_SOURCES := main.swift
OBJC_SOURCES := objc.m
CFLAGS_EXTRAS = -I$(BUILDDIR)/include
SWIFTFLAGS_EXTRAS = -I$(BUILDDIR)/include
LD_EXTRAS := libLibrary.dylib

include Makefile.rules

objc.o: libLibrary.dylib

libLibrary.dylib: Library.swift
	$(MAKE) -f $(MAKEFILE_RULES) \
		DYLIB_SWIFT_SOURCES=Library.swift \
		DYLIB_NAME=Library \
		SWIFTFLAGS_EXTRAS="-emit-objc-header-path $(BUILDDIR)/include/Library-Swift.h"
