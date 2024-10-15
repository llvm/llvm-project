SWIFT_SOURCES := main.swift
SWIFTFLAGS_EXTRAS = -I$(BUILDDIR)
LD_EXTRAS = -L$(BUILDDIR) -lFoo

all: libFoo.dylib $(EXE)

include Makefile.rules

libFoo.dylib: Foo.swift
	"$(MAKE)" MAKE_DSYM=NO VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/dylib.mk all
