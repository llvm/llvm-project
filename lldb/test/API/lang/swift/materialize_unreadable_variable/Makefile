SWIFT_SOURCES := main.swift

# Both libraries are resilient so that Unreadable is address-only in the
# consumer, matching the frame shapes this failure is reported from.
SWIFTFLAGS_EXTRAS = -I$(BUILDDIR) -enable-library-evolution
LD_EXTRAS = -L$(BUILDDIR) -lLib -lNoRefl

all: Lib.swiftmodule $(EXE)

include Makefile.rules

Lib.swiftmodule: $(SRCDIR)/Lib.swift NoRefl.swiftmodule
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES \
		DYLIB_NAME=Lib \
		DYLIB_SWIFT_SOURCES=Lib.swift \
		SWIFTFLAGS_EXTRAS="-I$(BUILDDIR) -enable-library-evolution" \
		LD_EXTRAS="-L$(BUILDDIR) -lNoRefl"

# -disable-reflection-metadata is what makes Opaque unreadable.
NoRefl.swiftmodule: $(SRCDIR)/NoRefl.swift
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES \
		DYLIB_NAME=NoRefl \
		DYLIB_SWIFT_SOURCES=NoRefl.swift \
		SWIFTFLAGS_EXTRAS="-enable-library-evolution -Xfrontend -disable-reflection-metadata"
