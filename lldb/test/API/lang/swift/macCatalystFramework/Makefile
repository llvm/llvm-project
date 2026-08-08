# A macCatalyst (arm64-apple-ios-macabi) executable, loading a plain
# macOS (arm64-apple-macosx) Swift framework.
C_SOURCES := main.c
LD_EXTRAS := -F. -framework Framework
CFLAGS_EXTRAS = -target $(ARCH)-apple-ios26.0-macabi

all: Framework.framework/Versions/A/Framework a.out

include Makefile.rules

# The framework is built for plain macOS -- note the macosx -target.
Framework.framework/Versions/A/Framework: $(SRCDIR)/Framework.swift
	touch Framework.h
	"$(MAKE)" -f $(MAKEFILE_RULES) \
	    MAKE_DSYM=YES \
		DYLIB_ONLY=YES \
		DYLIB_SWIFT_SOURCES=Framework.swift \
		MODULENAME=Framework \
		FRAMEWORK=Framework \
		FRAMEWORK_MODULES=Framework.swiftinterface \
		FRAMEWORK_HEADERS=Framework.h \
		SWIFTFLAGS_EXTRAS="-target $(ARCH)-apple-macosx26.0"
	$(call RM_F,$(BUILDDIR)/Framework.swiftmodule $(BUILDDIR)/Framework.swiftinterface)
	$(call RM_F,Framework.o Framework.h Framework.partial.swiftmodule)
