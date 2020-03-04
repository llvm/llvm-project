LEVEL = ../../../../make
DYLIB_ONLY := YES
DYLIB_NAME := $(BASENAME)
DYLIB_SWIFT_SOURCES := $(DYLIB_NAME).swift
SWIFT_OBJC_INTEROP := 1

include $(LEVEL)/Makefile.rules
