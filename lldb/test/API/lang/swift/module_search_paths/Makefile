SWIFT_SOURCES := main.swift
all: module-build

include Makefile.rules

# This is an extra dependency that does NOT get linked into the EXE.
module-build: \
              $(call tool_prereq,$(SWIFTC_PATH)) $(call tool_prereq,$(SWIFT_FRONTEND_PATH))
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		SWIFT_SOURCES=module.swift EXE=Module
