SWIFT_OBJC_INTEROP := 1
SWIFT_SOURCES := main.swift

all: module.modulemap conflict.h a.out

include Makefile.rules
SWIFTFLAGS += -Xcc -F$(SRCDIR) -I$(SRCDIR) -I$(BUILDDIR)

# Create a nonmodular include by removing the modulemap after building.
all:
	$(call RM,module.modulemap)

conflict.h: conflict.h.in
	$(call CP,$<,$@)

module.modulemap:
	$(call ECHO_TO_FILE,'module Baz { header "conflict.h" }',$@)
