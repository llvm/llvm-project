EXE = PlaygroundStub
SWIFT_SOURCES = PlaygroundStub.swift
# The real playground stub has no debug info and is written in C.
SWIFTFLAGS_EXTRAS = -gnone

# The deployment target we set is pre-ABI stability. The Swift driver will not
# point the RPATH at the system library. Do it manually.
LD_EXTRAS := -Xlinker -rpath -Xlinker /usr/lib/swift
LD_EXTRAS += -L. -lPlaygroundsRuntime -F. -framework AuxSources

PlaygroundStub: libPlaygroundsRuntime.dylib Dylib.framework AuxSources.framework

include Makefile.rules

libPlaygroundsRuntime.dylib: PlaygroundsRuntime.swift \
                             $(call tool_prereq,$(SWIFTC_PATH)) $(call tool_prereq,$(SWIFT_FRONTEND_PATH))
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_SWIFT_SOURCES=PlaygroundsRuntime.swift \
		DYLIB_NAME=PlaygroundsRuntime

Dylib.framework: Dylib.swift \
                 $(call tool_prereq,$(SWIFTC_PATH)) $(call tool_prereq,$(SWIFT_FRONTEND_PATH))
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		FRAMEWORK=Dylib \
		DYLIB_SWIFT_SOURCES=Dylib.swift \
		DYLIB_NAME=Dylib \
		SWIFTFLAGS_EXTRAS="-Xcc -DNEW_OPTION_FROM_DYLIB=1"

AuxSources.framework: AuxSources.swift \
                      $(call tool_prereq,$(SWIFTC_PATH)) $(call tool_prereq,$(SWIFT_FRONTEND_PATH))
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		FRAMEWORK=AuxSources \
		DYLIB_SWIFT_SOURCES=AuxSources.swift \
		DYLIB_NAME=AuxSources \
		SWIFTFLAGS_EXTRAS="-Xcc -DHAVE_AUXSOURCES=1"
