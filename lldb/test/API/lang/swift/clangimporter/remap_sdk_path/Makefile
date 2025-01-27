SWIFT_SOURCES := main.swift
SWIFTFLAGS_EXTRAS = -Xcc -I$(BUILD_SDK)/usr/include/net

include Makefile.rules

BUILD_SDK := $(BUILDDIR)/LocalSDK/$(shell basename $(SDKROOT))

$(EXE): $(SWIFT_SOURCES)
	rm -rf $(BUILDDIR)/LocalSDK
	echo "Symlinking iOS SDK into build directory as a fake macOS SDK"
	mkdir $(BUILDDIR)/LocalSDK
	ln -s $(SDKROOT) $(BUILD_SDK)
	echo "Building using SDK in build directory"
	"$(MAKE)" MAKE_DSYM=$(MAKE_DSYM) CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		VPATH=$(SRCDIR) -I $(SRCDIR) \
		SDKROOT=$(BUILD_SDK) \
		SWIFTFLAGS_EXTRAS="$(SWIFTFLAGS_EXTRAS)" \
		-f $(SRCDIR)/helper.mk clean main.swift.o a.swiftmodule
	echo "Sanity check that our SDK shenanigns worked"
	dwarfdump -r 0 $(BUILDDIR)/main.swift.o | egrep 'DW_AT_LLVM_i?sysroot' | grep -q LocalSDK
	echo "Linking with regular SDK (otherwise the linker complains)"
	"$(MAKE)" MAKE_DSYM=$(MAKE_DSYM) CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		VPATH=$(SRCDIR) -I $(SRCDIR) \
		SDKROOT=$(SDKROOT) \
		-f $(SRCDIR)/helper.mk all
	echo "Deleting build SDK"
	rm -rf $(BUILDDIR)/LocalSDK
