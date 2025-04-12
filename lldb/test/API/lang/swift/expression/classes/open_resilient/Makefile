SWIFT_SOURCES := main.swift
SWIFTFLAGS_EXTRAS := -I.
LD_EXTRAS = -lLibrary -L$(BUILDDIR)
all: libLibrary.dylib a.out

lib%.dylib: %.swift
	"$(MAKE)" MAKE_DSYM=NO DYLIB_ONLY=YES \
		DYLIB_HIDE_SWIFTMODULE=YES \
		DYLIB_NAME=$(shell basename $< .swift) \
		DYLIB_SWIFT_SOURCES=$(shell basename $<) \
                SWIFTFLAGS_EXTRAS=-enable-library-evolution \
		VPATH=$(SRCDIR) -I $(SRCDIR) -f $(MAKEFILE_RULES) all

include Makefile.rules
