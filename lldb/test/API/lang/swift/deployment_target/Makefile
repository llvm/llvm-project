all: libNewerTarget.dylib a.out dlopen_module

SWIFT_SOURCES=main.swift
SWIFTFLAGS_EXTRAS=-target $(ARCH)-apple-macosx14 -Xfrontend -disable-target-os-checking -I$(BUILDDIR)
LD_EXTRAS=-lNewerTarget -L$(BUILDDIR)

include Makefile.rules

# This test only works on macOS 15+.

a.out: main.swift libNewerTarget.dylib


dlopen_module: dlopen_module.m libNewerTarget.dylib \
               $(call tool_prereq,$(CC))
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		CXX= \
		OBJC_SOURCES=dlopen_module.m \
		EXE=dlopen_module \
		TRIPLE=$(ARCH)-apple-macosx14 \
		LD_EXTRAS="-framework Foundation"

lib%.dylib: %.swift \
            $(call tool_prereq,$(SWIFTC_PATH)) $(call tool_prereq,$(SWIFT_FRONTEND_PATH))
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES \
		DYLIB_NAME=$(patsubst lib%.dylib,%,$@) \
		DYLIB_SWIFT_SOURCES=$(patsubst lib%.dylib,%.swift,$@) \
		SWIFTFLAGS_EXTRAS="-target $(ARCH)-apple-macosx15"

clean::
	$(call RM_RF,*.swiftmodule *.swiftdoc *.dSYM *~ lib*.dylib a.out dlopen_module *.o)
