LEVEL = ../../../make

all: UnitTest.xctest/Contents/MacOS/test UnitTest.xctest/Contents/Info.plist

DYLIB_SWIFT_SOURCES := test.swift
DYLIB_NAME := test
DYLIB_ONLY := YES
include $(LEVEL)/Makefile.rules

UnitTest.xctest/Contents/MacOS/test: $(DYLIB_FILENAME) $(DSYM)
	mkdir -p $(BUILDDIR)/UnitTest.xctest/Contents/MacOS
	mv $(DYLIB_FILENAME) $(BUILDDIR)/UnitTest.xctest/Contents/MacOS/test
ifneq "$(MAKE_DSYM)" "NO"
	mv $(DSYM)/Contents/Resources/DWARF/$(DYLIB_FILENAME) \
	   $(DSYM)/Contents/Resources/DWARF/$(DYLIB_NAME)
	mv $(DSYM) $(BUILDDIR)/UnitTest.xctest/Contents/MacOS/test.dSYM
endif

UnitTest.xctest/Contents/Info.plist: Info.plist
	cp $< $(BUILDDIR)/UnitTest.xctest/Contents/

clean::
	rm -rf *.o *.dSYM *.dylib *.swiftdoc *.swiftmodule *.xctest xctest
