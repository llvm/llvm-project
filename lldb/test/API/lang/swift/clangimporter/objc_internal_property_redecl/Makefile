SWIFT_SOURCES = main.swift
SWIFT_OBJC_INTEROP := 1

all: TestFramework.framework a.out

include Makefile.rules

SWIFTFLAGS_EXTRAS = \
  -F$(BUILDDIR) -framework TestFramework \
  -Xlinker -rpath -Xlinker $(BUILDDIR)

TestFramework.framework: TestFramework.swift TestFramework.h \
                         TestFramework_Private.h \
                         module.modulemap module.private.modulemap \
                         $(call tool_prereq,$(SWIFTC_PATH)) $(call tool_prereq,$(SWIFT_FRONTEND_PATH))
	"$(MAKE)" -f $(MAKEFILE_RULES) -C $(BUILDDIR) VPATH=$(VPATH) \
		MAKE_DSYM=NO CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		DYLIB_NAME=TestFramework \
		DYLIB_ONLY=YES \
		DYLIB_SWIFT_SOURCES=TestFramework.swift \
		SWIFT_OBJC_INTEROP=1 \
		SWIFT_OBJC_HEADER=TestFramework-Swift.h \
		FRAMEWORK=TestFramework \
		FRAMEWORK_HEADERS="$(SRCDIR)/TestFramework.h $(BUILDDIR)/TestFramework-Swift.h" \
		FRAMEWORK_MODULES="$(SRCDIR)/module.modulemap $(SRCDIR)/module.private.modulemap" \
		FRAMEWORK_PRIVATE_HEADERS="$(SRCDIR)/TestFramework_Private.h" \
		DYLIB_INSTALL_NAME="@executable_path/TestFramework.framework/Versions/A/TestFramework"
	$(call LN_SF,$(BUILDDIR)/TestFramework.framework/TestFramework,$(BUILDDIR)/TestFramework)
