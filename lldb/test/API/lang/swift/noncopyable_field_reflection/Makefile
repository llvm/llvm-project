SWIFT_SOURCES := main.swift

# Pin the deployment target below macOS 27 (NoncopyableReflectionSafety).
# This forces IRGen to emit the noncopyable field's reflection type ref as
# an accessor-function symbolic reference.
DEPLOYMENT_TARGET = -target $(ARCH)-apple-macosx15.0

SWIFTFLAGS_EXTRAS = -I. -enable-library-evolution $(DEPLOYMENT_TARGET)

LD_EXTRAS = -L. -Xlinker -rpath -Xlinker $(BUILDDIR) -lLib

all: dylib $(EXE)
.PHONY: dylib

include Makefile.rules
dylib: lib.swift \
       $(call tool_prereq,$(SWIFTC_PATH)) $(call tool_prereq,$(SWIFT_FRONTEND_PATH))
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		MAKE_DSYM=YES DYLIB_ONLY=YES DYLIB_NAME=Lib \
		DYLIB_SWIFT_SOURCES="lib.swift" \
		SWIFTFLAGS_EXTRAS="-I$(BUILDDIR) -enable-library-evolution $(DEPLOYMENT_TARGET)"
