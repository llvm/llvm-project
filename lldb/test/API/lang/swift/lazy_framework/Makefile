SWIFT_SOURCES := main.swift

SWIFTFLAGS_EXTRAS = -F $(BUILDDIR)

all: Lazy.framework $(EXE)

include Makefile.rules

Lazy.framework: $(SRCDIR)/Lazy.swift \
                $(call tool_prereq,$(SWIFTC_PATH)) $(call tool_prereq,$(SWIFT_FRONTEND_PATH))
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_NAME=Lazy \
		DYLIB_SWIFT_SOURCES=Lazy.swift \
		DYLIB_MODULENAME=Lazy \
		FRAMEWORK=Lazy
	$(call RM_F,$(BUILDDIR)/Lazy.swiftmodule $(BUILDDIR)/Lazy.swiftinterface)
