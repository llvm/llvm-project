ifeq "$(OS)" "Darwin"
	OS_SWIFTFLAGS := -target $(ARCH)-apple-macosx13.0
endif

SWIFT_SOURCES := main.swift
SWIFTFLAGS_EXTRAS := -enable-bare-slash-regex $(OS_SWIFTFLAGS)

include Makefile.rules
