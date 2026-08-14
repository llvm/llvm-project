export SWIFT_ENABLE_EXPLICIT_MODULES := NO
SWIFT_SOURCES := main.swift
SWIFT_BRIDGING_HEADER := bridging.h
SWIFT_PRECOMPILE_BRIDGING_HEADER := YES
LD_EXTRAS := -L. -linvisible
SWIFTFLAGS_EXTRAS := -I.

all: libinvisible.dylib $(EXE)

include Makefile.rules

libinvisible.dylib: Invisible.swift \
                    $(call tool_prereq,$(SWIFTC_PATH)) $(call tool_prereq,$(SWIFT_FRONTEND_PATH))
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		MAKE_DSYM=YES DYLIB_ONLY=YES DYLIB_NAME=Invisible \
		DYLIB_SWIFT_SOURCES="Invisible.swift" \
		SWIFTFLAGS_EXTRAS=-I$(BUILDDIR)
