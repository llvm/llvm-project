SWIFT_SOURCES := main.swift
SWIFTFLAGS_EXTRAS = -F$(BUILDDIR) -Xlinker -rpath -Xlinker $(BUILDDIR)

all: OtherArch.framework/Versions/A/OtherArch $(EXE)

include Makefile.rules

OtherArch.framework/Versions/A/OtherArch: $(SRCDIR)/OtherArch.swift
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
