SWIFT_SOURCES := main.swift
SWIFT_BRIDGING_HEADER := bridgingA.h
SWIFTFLAGS_EXTRAS = -auto-bridging-header-chaining -Xcc -I. -I. \
  -F$(BUILDDIR) -framework B \
  -Xlinker -rpath -Xlinker $(BUILDDIR)

all: secret.h B.framework $(EXE)

secret.h:
	echo "/* will be removed to ensure we're loading the PCH */" >$@

include Makefile.rules

B.framework: \
             $(call tool_prereq,$(SWIFTC_PATH)) $(call tool_prereq,$(SWIFT_FRONTEND_PATH))
	$(MAKE) MAKE_DSYM=$(MAKE_DSYM) CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		VPATH=$(SRCDIR) -I $(SRCDIR) SRCDIR=$(SRCDIR) \
		-f $(THIS_FILE_DIR)/Makefile.rules \
		DYLIB_SWIFT_SOURCES=B.swift \
		SWIFT_BRIDGING_HEADER=bridgingB.h \
		SWIFT_PRECOMPILE_BRIDGING_HEADER=YES \
		SWIFTFLAGS_EXTRAS="-Xcc -I." \
		DYLIB_NAME=B \
		FRAMEWORK=B \
		DYLIB_ONLY=YES \
		DYLIB_INSTALL_NAME="@executable_path/B" \
		all
	$(call LN_SF,$(BUILDDIR)/B.framework/B,$(BUILDDIR)/B)
