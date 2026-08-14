export SWIFT_ENABLE_EXPLICIT_MODULES := NO
SWIFT_SOURCES := main.swift
SWIFTFLAGS_EXTRAS = -I$(BUILDDIR)
LD_EXTRAS = -L$(BUILDDIR) -lDirect

# rpath is a Unix loader concept rejected by lld-link.
RPATH_HIDDEN = -Xlinker -rpath -Xlinker $(BUILDDIR)/hidden
ifeq "$(OS)" "Windows_NT"
RPATH_HIDDEN =
endif

all: libDirect $(EXE)
.phony: libDirect libHidden

include Makefile.rules

libHidden: $(SRCDIR)/hidden/Indirect.swift \
           $(call tool_prereq,$(SWIFTC_PATH)) $(call tool_prereq,$(SWIFT_FRONTEND_PATH))
	$(call MKDIR_P,$(BUILDDIR)/hidden)
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		-C $(BUILDDIR)/hidden VPATH=$(SRCDIR)/hidden \
		DYLIB_ONLY=YES \
		DYLIB_NAME=Indirect \
		DYLIB_SWIFT_SOURCES=Indirect.swift \
		DYLIB_MODULENAME=Indirect \
		SWIFTFLAGS_EXTRAS="-gnone" \
		MAKE_DSYM=NO

libDirect: $(SRCDIR)/Direct.swift libHidden \
           $(call tool_prereq,$(SWIFTC_PATH)) $(call tool_prereq,$(SWIFT_FRONTEND_PATH))
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES \
		DYLIB_NAME=Direct \
		DYLIB_SWIFT_SOURCES=Direct.swift \
		DYLIB_MODULENAME=Direct \
		SWIFTFLAGS_EXTRAS="-I$(BUILDDIR)/hidden" \
		LD_EXTRAS="-L$(BUILDDIR)/hidden -lIndirect $(RPATH_HIDDEN)" \
		MAKE_DSYM=NO

