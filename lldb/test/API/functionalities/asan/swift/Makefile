
SWIFT_SOURCES := main.swift

# Note the lazy variable setting - needs to use CLANG_RT_DIR, defined in Makefile.rules
LD_EXTRAS = -lclang_rt.asan_osx_dynamic -L$(CLANG_RT_DIR)

include Makefile.rules

SWIFTFLAGS += -sanitize=address
