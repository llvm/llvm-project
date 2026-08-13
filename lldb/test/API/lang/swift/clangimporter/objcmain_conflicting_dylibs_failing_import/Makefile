export SWIFT_ENABLE_EXPLICIT_MODULES := NO
OBJC_SOURCES := main.m
LD_EXTRAS = -lFoo -lBar -L$(BUILDDIR)
CFLAGS_EXTRAS = -I$(BUILDDIR) -fmodules

# This test builds an Objective-C main program that imports two Swift
# .dylibs with conflicting ClangImporter search paths.

all: libFoo.dylib libBar.dylib a.out

include Makefile.rules

lib%.dylib: %.swift \
            $(call tool_prereq,$(SWIFTC_PATH)) $(call tool_prereq,$(SWIFT_FRONTEND_PATH))
	"$(MAKE)" MAKE_DSYM=YES CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		BASENAME=$(shell basename $< .swift) \
		VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/dylib.mk all

clean::
	$(call RM_RF,*.swiftmodule *.swiftdoc *.dSYM *~ lib*.dylib a.out *.o)
