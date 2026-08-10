SWIFT_SOURCES := main.swift
SWIFT_ENABLE_COMPILE_CACHE := YES
# Without a .swiftmodule next to the binary, the SwiftASTContext has to load
# the module out of the CAS, which is the reference we want released.
HIDE_SWIFTMODULE := YES
SWIFTFLAGS_EXTRAS = -I$(SRCDIR)

all: a.out cas-config

include Makefile.rules

cas-config:
	echo "{\"CASPath\": \"$(BUILDDIR)/cas\"}" > .cas-config
