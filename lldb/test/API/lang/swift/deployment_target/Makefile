all: libNewerTarget.dylib a.out dlopen_module

SWIFT_SOURCES=main.swift
SWIFTFLAGS_EXTRAS=-target x86_64-apple-macosx10.10 -Xfrontend -disable-target-os-checking -I$(BUILDDIR)
LD_EXTRAS=-lNewerTarget -L$(BUILDDIR)

include Makefile.rules

# This test only works on macOS 10.11+.

a.out: main.swift libNewerTarget.dylib

dlopen_module: dlopen_module.m libNewerTarget.dylib
	$(MAKE) -f $(MAKEFILE_RULES) \
		CXX= \
		OBJC_SOURCES=dlopen_module.m \
		EXE=dlopen_module \
		LD_EXTRAS="-framework Foundation -mmacos-version-min=10.8"

lib%.dylib: %.swift
	$(MAKE) -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES \
		DYLIB_NAME=$(patsubst lib%.dylib,%,$@) \
		DYLIB_SWIFT_SOURCES=$(patsubst lib%.dylib,%.swift,$@) \
		SWIFTFLAGS_EXTRAS="-target x86_64-apple-macosx10.11"

clean::
	rm -rf *.swiftmodule *.swiftdoc *.dSYM *~ lib*.dylib a.out dlopen_module *.o
