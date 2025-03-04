all: Dylib.framework

Dylib.framework: Dylib.swift
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		FRAMEWORK=Dylib \
		DYLIB_SWIFT_SOURCES=Dylib.swift \
		DYLIB_NAME=Dylib \
		SWIFTFLAGS_EXTRAS="-Xcc -DNEW_OPTION_FROM_DYLIB=1" \
		MAKE_DSYM=YES

include ../Makefile.common
