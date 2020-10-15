EXE = PlaygroundStub
SWIFT_SOURCES = PlaygroundStub.swift

# The deployment target we set is pre-ABI stability. The Swift driver will not
# point the RPATH at the system library. Do it manually.
LD_EXTRAS := -Xlinker -rpath -Xlinker /usr/lib/swift

PlaygroundStub: libPlaygroundsRuntime.dylib

include Makefile.rules

libPlaygroundsRuntime.dylib: PlaygroundsRuntime.swift
	$(MAKE) -f $(MAKEFILE_RULES) \
		DYLIB_SWIFT_SOURCES=PlaygroundsRuntime.swift \
		DYLIB_NAME=PlaygroundsRuntime

