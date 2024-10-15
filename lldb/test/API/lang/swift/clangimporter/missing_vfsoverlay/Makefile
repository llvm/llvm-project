SWIFT_SOURCES := main.swift
SWIFT_OBJC_INTEROP := 1
SWIFTFLAGS_EXTRAS = -I$(BUILDDIR)
LD_EXTRAS = -L$(BUILDDIR) -lFoo

all: libFoo.dylib $(EXE)

include Makefile.rules

OVERLAY := $(BUILDDIR)/overlay.yaml
lib%.dylib: %.swift
	rm -f $(OVERLAY)
	echo "{ 'version': 0, 'roots': [] }" >$(OVERLAY)
	"$(MAKE)" MAKE_DSYM=$(MAKE_DSYM) CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		BASENAME=$(shell basename $< .swift) \
		SWIFTFLAGS_EXTRAS="-Xcc -ivfsoverlay -Xcc $(OVERLAY)" \
		VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/dylib.mk all
	rm -f $(OVERLAY)

clean::
	"$(MAKE)" MAKE_DSYM=$(MAKE_DSYM) CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		BASENAME=$(shell basename $< .swift) \
		VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/dylib.mk clean
