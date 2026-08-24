all: Dylib.framework

# Prerequisite lists are expanded as the rule is read, so any rule using
# tool_prereq has to come after the include that defines it.
include ../Makefile.common

Dylib.framework: Dylib.swift \
                 $(call tool_prereq,$(SWIFTC_PATH)) $(call tool_prereq,$(SWIFT_FRONTEND_PATH))
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		FRAMEWORK=Dylib \
		DYLIB_SWIFT_SOURCES=Dylib.swift \
		DYLIB_NAME=Dylib \
		SWIFTFLAGS_EXTRAS="-Xcc -DNEW_OPTION_FROM_DYLIB=1" \
		MAKE_DSYM=YES
