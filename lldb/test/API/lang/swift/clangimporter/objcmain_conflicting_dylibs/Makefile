SWIFT_OBJC_INTEROP := 1

# This test builds an Objective-C main program that imports two Swift
# .dylibs with conflicting ClangImporter search paths.

include Makefile.rules

a.out: main.m libFoo.dylib libBar.dylib
	$(CC) $(CFLAGS) $(MANDATORY_MODULE_BUILD_CFLAGS) -I$(shell pwd) -lFoo -lBar -L$(shell pwd) -o $@ $<
ifneq "$(CODESIGN)" ""
	$(CODESIGN) -s - "$@"
endif

lib%.dylib: %.swift
	$(MAKE) MAKE_DSYM=YES \
		DYLIB_NAME=$(shell basename $< .swift) \
		VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/dylib.mk all

clean::
	rm -rf *.swiftmodule *.swiftdoc *.dSYM *~ lib*.dylib a.out *.o
