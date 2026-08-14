SWIFT_SOURCES := main.swift
MODULENAME = Main
SWIFTFLAGS_EXTRAS = -I$(BUILDDIR)
LD_EXTRAS = -L$(BUILDDIR) -lA -lB

all: A.swiftmodule $(EXE)

include Makefile.rules

A.swiftmodule: $(SRCDIR)/A.swift B.swiftmodule \
               $(call tool_prereq,$(SWIFTC_PATH)) $(call tool_prereq,$(SWIFT_FRONTEND_PATH))
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES \
		DYLIB_NAME=A \
		DYLIB_SWIFT_SOURCES=A.swift \
		DYLIB_MODULENAME=A \
		SWIFTFLAGS_EXTRAS=-I$(BUILDDIR) \
		LD_EXTRAS="-L$(BUILDDIR) -lB"

B.swiftmodule: $(SRCDIR)/B.swift \
               $(call tool_prereq,$(SWIFTC_PATH)) $(call tool_prereq,$(SWIFT_FRONTEND_PATH))
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES \
		DYLIB_NAME=B \
		DYLIB_SWIFT_SOURCES=B.swift \
		DYLIB_MODULENAME=B
