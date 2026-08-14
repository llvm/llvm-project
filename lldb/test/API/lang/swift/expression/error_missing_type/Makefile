export SWIFT_ENABLE_EXPLICIT_MODULES := NO
SWIFT_SOURCES := main.swift
SWIFTFLAGS_EXTRAS := -I.
LD_EXTRAS = -lLibrary -L$(BUILDDIR)
all: libLibrary.dylib a.out

lib%.dylib: %.swift \
            $(call tool_prereq,$(SWIFTC_PATH)) $(call tool_prereq,$(SWIFT_FRONTEND_PATH))
	"$(MAKE)" MAKE_DSYM=NO DYLIB_ONLY=YES \
		DYLIB_HIDE_SWIFTMODULE=YES \
		DYLIB_NAME=$(shell basename $< .swift) \
		DYLIB_SWIFT_SOURCES=$(shell basename $<) \
		VPATH=$(SRCDIR) -I $(SRCDIR) -f $(MAKEFILE_RULES) all
	$(call RM_F,Library.private.swiftinterface)

include Makefile.rules
