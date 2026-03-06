SWIFT_SOURCES := main.swift
SWIFT_OBJC_INTEROP := 1
SWIFTFLAGS_EXTRAS = -I$(BUILDDIR)
LD_EXTRAS = -L$(BUILDDIR) -lFoo

all: libFoo.dylib $(EXE)

include Makefile.rules

OVERLAY_CONTENT := { "version": 0, "roots": [] }
OVERLAY := $(BUILDDIR)/overlay.yaml

lib%.dylib: %.swift
	$(call RM_F,$(OVERLAY))
	$(call ECHO_TO_FILE,'$(OVERLAY_CONTENT)',$(OVERLAY))
	"$(MAKE)" MAKE_DSYM=$(MAKE_DSYM) CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		BASENAME=$(shell basename $< .swift) \
		SWIFTFLAGS_EXTRAS="-Xcc -ivfsoverlay -Xcc $(OVERLAY)" \
		VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/dylib.mk all
	$(call RM_F,$(OVERLAY))

clean::
	"$(MAKE)" MAKE_DSYM=$(MAKE_DSYM) CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		BASENAME=$(shell basename $< .swift) \
		VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/dylib.mk clean
