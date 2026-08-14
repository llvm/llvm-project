SRCDIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
SWIFT_OBJC_INTEROP := 1

# This test builds an Objective-C main program that imports two Swift
# .dylibs with conflicting -D macro definitions.

include Makefile.rules

a.out: main.m libFoo.dylib libBar.dylib \
       $(call tool_prereq,$(CC))
	$(CC) $(CFLAGS) $(MANDATORY_MODULE_BUILD_CFLAGS) -I$(shell pwd) -lFoo -lBar -L$(shell pwd) -o $@ $<
ifneq "$(CODESIGN)" ""
	$(CODESIGN) -s - "$@"
endif


lib%.dylib: %.swift \
            $(call tool_prereq,$(SWIFTC_PATH)) $(call tool_prereq,$(SWIFT_FRONTEND_PATH))
	"$(MAKE)" MAKE_DSYM=$(MAKE_DSYM) CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		BASENAME=$(shell basename $< .swift) \
		VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/dylib.mk all

clean::
	$(call RM_RF,*.swiftmodule *.swiftdoc *.dSYM *~ lib*.dylib a.out *.o)
