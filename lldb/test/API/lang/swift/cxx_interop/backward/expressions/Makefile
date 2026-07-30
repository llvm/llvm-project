SWIFT_ENABLE_EXPLICIT_MODULES := NO
SWIFT_CXX_HEADER := swift-types.h
SWIFT_SOURCES := swift-types.swift
CXX_SOURCES := main.cpp
SWIFT_CXX_INTEROP := 1
SWIFTFLAGS_EXTRAS = -Xcc -I$(SRCDIR) -parse-as-library
# On Windows bare -g emits CodeView, but this test (and the Swift side) use
# DWARF, so lldb could not resolve source breakpoints in main.cpp. Match the
# harness default (-gdwarf) there; leave other platforms unchanged.
ifeq "$(OS)" "Windows_NT"
CFLAGS = -I. -gdwarf
else
CFLAGS = -I. -g
endif
include Makefile.rules
