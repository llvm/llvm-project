
SWIFT_SOURCES := main.swift

# Note the lazy variable setting - needs to use CLANG_RT_DIR, defined in Makefile.rules
asan: LD_EXTRAS = -lclang_rt.asan_osx_dynamic -L$(CLANG_RT_DIR)
asan: SWIFTFLAGS += -sanitize=address
asan: all

libsanitizers: SWIFTFLAGS += -sanitize=address -sanitize-stable-abi
libsanitizers: all

include Makefile.rules
