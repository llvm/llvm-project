SWIFT_SOURCES := main.swift
SWIFTFLAGS_EXTRAS = -Xcc -I$(BUILD_SDK)/usr/include/net

include Makefile.rules

BUILD_SDK := $(BUILDDIR)/LocalSDK/$(shell basename $(SDKROOT))

$(EXE): $(SWIFT_SOURCES)
	$(call RM_RF,$(BUILDDIR)/LocalSDK)
	$(call ECHO,"Symlinking iOS SDK into build directory as a fake macOS SDK")
	$(call MKDIR_P,$(BUILDDIR)/LocalSDK)
	$(call LN_SF,$(SDKROOT),$(BUILD_SDK))
	$(call ECHO,"Building using SDK in build directory")
	"$(MAKE)" MAKE_DSYM=$(MAKE_DSYM) CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		VPATH=$(SRCDIR) -I $(SRCDIR) \
		SDKROOT=$(BUILD_SDK) \
		SWIFTFLAGS_EXTRAS="$(SWIFTFLAGS_EXTRAS)" \
		-f $(SRCDIR)/helper.mk clean main.swift.o a.swiftmodule
	$(call ECHO,"Sanity check that our SDK shenanigns worked")
	dwarfdump -r 0 $(BUILDDIR)/main.swift.o | egrep 'DW_AT_LLVM_i?sysroot' | grep -q LocalSDK
	$(call ECHO,"Linking with regular SDK (otherwise the linker complains)")
	"$(MAKE)" MAKE_DSYM=$(MAKE_DSYM) CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		VPATH=$(SRCDIR) -I $(SRCDIR) \
		SDKROOT=$(SDKROOT) \
		-f $(SRCDIR)/helper.mk all
	$(call ECHO,"Deleting build SDK")
	$(call RM_RF,$(BUILDDIR)/LocalSDK)
