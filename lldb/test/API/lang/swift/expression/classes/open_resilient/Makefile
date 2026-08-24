SWIFT_SOURCES := main.swift
SWIFTFLAGS_EXTRAS := -I.
LD_EXTRAS = -lLibrary -L$(BUILDDIR)
all: libLibrary.dylib a.out

# Prerequisite lists are expanded as the rule is read, so any rule using
# tool_prereq has to come after the include that defines it.
include Makefile.rules

lib%.dylib: %.swift \
            $(call tool_prereq,$(SWIFTC_PATH)) $(call tool_prereq,$(SWIFT_FRONTEND_PATH))
	"$(MAKE)" MAKE_DSYM=NO DYLIB_ONLY=YES \
		DYLIB_HIDE_SWIFTMODULE=YES \
		DYLIB_NAME=$(shell basename $< .swift) \
		DYLIB_SWIFT_SOURCES=$(shell basename $<) \
                SWIFTFLAGS_EXTRAS=-enable-library-evolution \
		VPATH=$(SRCDIR) -I $(SRCDIR) -f $(MAKEFILE_RULES) all
