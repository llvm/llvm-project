SWIFT_SOURCES := main.swift
SWIFT_OBJC_INTEROP := 1
SWIFTFLAGS_EXTRAS = -I$(BUILDDIR)
LD_EXTRAS = -L$(BUILDDIR) -lFoo

all: libFoo.dylib $(EXE)

include Makefile.rules

OVERLAY := $(BUILDDIR)/overlay.yaml
lib%.dylib: %.swift
	echo "struct S {};">$(BUILDDIR)/header.h
	"$(MAKE)" MAKE_DSYM=$(MAKE_DSYM) CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		BASENAME=$(shell basename $< .swift) \
	        DISABLE_SWIFT_INTERFACE=YES \
		SWIFTFLAGS_EXTRAS="-import-objc-header header.h -enable-bridging-pch -pch-output-dir $(BUILDDIR)" \
                VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/dylib.mk all
	rm -f $(OVERLAY)
	rm header*.pch

clean::
	"$(MAKE)" MAKE_DSYM=$(MAKE_DSYM) CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		BASENAME=$(shell basename $< .swift) \
		VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/dylib.mk clean
