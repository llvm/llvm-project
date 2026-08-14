export SWIFT_ENABLE_EXPLICIT_MODULES := NO
SWIFT_SOURCES := main.swift

SWIFTFLAGS_EXTRAS = -F $(BUILDDIR) -framework Direct -Xlinker -rpath -Xlinker $(BUILDDIR) -Fother_secret_path

all: Direct.framework $(EXE)

include Makefile.rules

Discovered.framework: Discovered.h \
                      $(call tool_prereq,$(CC))
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES \
		DYLIB_NAME=Discovered \
		DYLIB_OBJC_SOURCES=Discovered.m \
		FRAMEWORK_HEADERS=$(SRCDIR)/Discovered.h \
		FRAMEWORK_MODULES=$(SRCDIR)/module.modulemap \
		FRAMEWORK=Discovered

Direct.framework: $(SRCDIR)/Direct.swift.in Discovered.framework \
                  $(call tool_prereq,$(SWIFTC_PATH)) $(call tool_prereq,$(SWIFT_FRONTEND_PATH))
	$(call MKDIR_P,$(BUILDDIR)/secret_path)
	$(call CP,$<,$(BUILDDIR)/Direct.swift)
	# Clear the destination: this rule re-runs whenever the toolchain changes,
	# and mv refuses to replace a non-empty directory.
	$(call RM_RF,$(BUILDDIR)/secret_path/Discovered.framework)
	mv Discovered.framework $(BUILDDIR)/secret_path
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_NAME=Direct \
		DYLIB_SWIFT_SOURCES=Direct.swift \
		DYLIB_MODULENAME=Direct \
		FRAMEWORK=Direct \
		SWIFTFLAGS_EXTRAS="-F$(BUILDDIR)/secret_path"
	$(call RM_F,$(BUILDDIR)/Direct.swiftmodule $(BUILDDIR)/Direct.swiftinterface $(BUILDDIR)/Direct.swift)
