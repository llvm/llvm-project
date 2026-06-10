SWIFT_ENABLE_EXPLICIT_MODULES := NO
SWIFT_CXX_HEADER := swift-types.h
SWIFT_SOURCES := swift-types.swift
CXX_SOURCES := main.cpp
SWIFT_CXX_INTEROP := 1
SWIFTFLAGS_EXTRAS = -Xcc -I$(SRCDIR) -parse-as-library
ifeq "$(OS)" "Windows_NT"
CFLAGS_EXTRAS := -I. -gdwarf
else
CFLAGS = -I. -g
endif
include Makefile.rules
