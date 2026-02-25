SWIFT_SOURCES := main.swift
SWIFT_ENABLE_EXPLICIT_MODULES := YES
SWIFT_ENABLE_COMPILE_CACHE := YES
HIDE_SWIFTMODULE := YES
SWIFTFLAGS_EXTRAS = -I$(SRCDIR) -I/TEST_DIR -F/FRAMEWORK_DIR

all: a.out cas-config

include Makefile.rules

cas-config:
	echo "{\"CASPath\": \"$(BUILDDIR)/cas\"}" > .cas-config
