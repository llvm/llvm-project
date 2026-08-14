SWIFT_SOURCES := main.swift
SWIFTFLAGS_EXTRAS := -I.
LD_EXTRAS = -L. -Xlinker -rpath -Xlinker $(BUILDDIR) -lLib

all: clean dylib a.out

include Makefile.rules

# DYLIB_HIDE_SWIFTMODULE together with no reflection metadata makes it
# impossible to lay out `self`.
.PHONY: dylib
dylib: Lib.swift \
       $(call tool_prereq,$(SWIFTC_PATH)) $(call tool_prereq,$(SWIFT_FRONTEND_PATH))
	"$(MAKE)" -f $(MAKEFILE_RULES) -C $(BUILDDIR) VPATH=$(SRCDIR) \
		CC=$(CC) SWIFTC=$(SWIFTC) ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		DYLIB_HIDE_SWIFTMODULE=YES \
		DYLIB_ONLY=YES \
		DYLIB_NAME=Lib \
		DYLIB_SWIFT_SOURCES=Lib.swift \
		SWIFTFLAGS_EXTRAS="-Xfrontend -disable-reflection-metadata -enable-library-evolution"
