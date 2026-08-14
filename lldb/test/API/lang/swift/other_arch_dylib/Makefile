export SWIFT_ENABLE_EXPLICIT_MODULES := NO
SWIFT_SOURCES := main.swift
SWIFTFLAGS_EXTRAS = -F$(BUILDDIR) -Xlinker -rpath -Xlinker $(BUILDDIR)

# OtherArch.framework below is built for macOS 15, and a module cannot be
# imported by code with a lower deployment target.
SWIFT_DEPLOYMENT_TARGET := 26

all: OtherArch.framework/Versions/A/OtherArch $(EXE)

include Makefile.rules

OtherArch.framework/Versions/A/OtherArch: $(SRCDIR)/OtherArch.swift \
                                          $(call tool_prereq,$(SWIFTC_PATH)) $(call tool_prereq,$(SWIFT_FRONTEND_PATH))
	touch OtherArch.h
	"$(MAKE)" -f $(MAKEFILE_RULES) \
	    MAKE_DSYM=YES \
		DYLIB_ONLY=YES \
		DYLIB_SWIFT_SOURCES=OtherArch.swift \
		MODULENAME=OtherArch \
		FRAMEWORK=OtherArch \
		FRAMEWORK_MODULES=OtherArch.swiftinterface \
		FRAMEWORK_HEADERS=OtherArch.h \
		SWIFTFLAGS_EXTRAS="-target arm64e-apple-macos15"
	$(call RM_F,$(BUILDDIR)/OtherArch.swiftmodule $(BUILDDIR)/OtherArch.swiftinterface)
	$(call RM_F,OtherArch.o OtherArch.h OtherArch.partial.swiftmodule)
