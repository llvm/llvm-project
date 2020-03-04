SWIFT_SOURCES := main.swift

all: module-build

include Makefile.rules

# This is an extra dependency that does NOT get linked into the EXE.
module-build:
	$(MAKE) -f $(MAKEFILE_RULES) \
		SWIFT_SOURCES=module.swift EXE=Module
