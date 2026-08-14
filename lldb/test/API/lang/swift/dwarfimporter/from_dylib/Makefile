export SWIFT_ENABLE_EXPLICIT_MODULES := NO
SWIFT_SOURCES := main.swift
SWIFTFLAGS_EXTRAS = -I$(BUILDDIR)
LD_EXTRAS = -L$(BUILDDIR) -lFoo

all: libFoo.dylib $(EXE)

include Makefile.rules

libFoo.dylib: Foo.swift \
              $(call tool_prereq,$(SWIFTC_PATH)) $(call tool_prereq,$(SWIFT_FRONTEND_PATH))
	"$(MAKE)" MAKE_DSYM=NO VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/dylib.mk all
