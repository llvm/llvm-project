SWIFT_OBJC_INTEROP := 1

include Makefile.rules

BUILDDIR := $(realpath $(shell pwd))
BOTDIR := $(BUILDDIR)/buildbot
USERDIR := $(BUILDDIR)/user

a.out: main.m libFoo.dylib
	$(CC) $(CFLAGS) $(MANDATORY_MODULE_BUILD_CFLAGS) -I$(shell pwd) -lFoo -L$(shell pwd) -o $@ $<
ifneq "$(CODESIGN)" ""
	$(CODESIGN) -s - "$@"
endif
	mv $(BOTDIR) $(USERDIR)

libFoo.dylib: Foo.swift
	$(call MKDIR_P,$(BOTDIR))
	$(call CP_R,$(SRCDIR)/Foo,$(BOTDIR)/)
	"$(MAKE)" MAKE_DSYM=YES CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		BASENAME=$(shell basename $< .swift) \
		BOTDIR=$(BOTDIR) \
		VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/dylib.mk all
	UUID=$$(dwarfdump --uuid "$@.dSYM" | awk '{ print $$2 }') ; \
	$(call ECHO,"UUID = $$UUID") \
	PLIST="$@.dSYM/Contents/Resources/$$UUID.plist" ; \
	$(call ECHO,"Generating $$PLIST ...") \
	$(call ECHO_TO_FILE,'<?xml version="1.0" encoding="UTF-8"?>',$$PLIST) \
	$(call ECHO_APPEND_FILE,'<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">',$$PLIST) \
	$(call ECHO_APPEND_FILE,'<plist version="1.0">',$$PLIST) \
	$(call ECHO_APPEND_FILE,'<dict>',$$PLIST) \
	$(call ECHO_APPEND_FILE,'  <key>DBGSourcePathRemapping</key>',$$PLIST) \
	$(call ECHO_APPEND_FILE,'  <dict>',$$PLIST) \
	$(call ECHO_APPEND_FILE,'    <key>$(BOTDIR)</key>',$$PLIST) \
	$(call ECHO_APPEND_FILE,'    <string>$(USERDIR)</string>',$$PLIST) \
	$(call ECHO_APPEND_FILE,'    <key>/nonexisting-rootdir</key>',$$PLIST) \
	$(call ECHO_APPEND_FILE,'    <string>$(USERDIR)</string>',$$PLIST) \
	$(call ECHO_APPEND_FILE,'  </dict>',$$PLIST) \
	$(call ECHO_APPEND_FILE,'</dict>',$$PLIST) \
	$(call ECHO_APPEND_FILE,'</plist>',$$PLIST) \

clean::
	$(call RM_RF,$(USERDIR) $(call RM_RF,*.swiftmodule *.swiftdoc *.dSYM *~ lib*.dylib a.out *.o))
