
SWIFT_SOURCES := main.swift

# Pick the matching asan runtime variant for the test's target platform.
# CLANG_RT_DIR is defined in Makefile.rules and resolves to the directory
# containing the platform-specific clang_rt asan dylibs.
ASAN_RT_NAME := asan_osx_dynamic
ifneq "$(findstring -apple-ios,$(TARGET_SWIFTFLAGS))" ""
  ifneq "$(findstring simulator,$(TARGET_SWIFTFLAGS))" ""
    ASAN_RT_NAME := asan_iossim_dynamic
  else
    ASAN_RT_NAME := asan_ios_dynamic
  endif
else ifneq "$(findstring -apple-tvos,$(TARGET_SWIFTFLAGS))" ""
  ifneq "$(findstring simulator,$(TARGET_SWIFTFLAGS))" ""
    ASAN_RT_NAME := asan_tvossim_dynamic
  else
    ASAN_RT_NAME := asan_tvos_dynamic
  endif
else ifneq "$(findstring -apple-watchos,$(TARGET_SWIFTFLAGS))" ""
  ifneq "$(findstring simulator,$(TARGET_SWIFTFLAGS))" ""
    ASAN_RT_NAME := asan_watchossim_dynamic
  else
    ASAN_RT_NAME := asan_watchos_dynamic
  endif
else ifneq "$(findstring -apple-xros,$(TARGET_SWIFTFLAGS))" ""
  ifneq "$(findstring simulator,$(TARGET_SWIFTFLAGS))" ""
    ASAN_RT_NAME := asan_xrossim_dynamic
  else
    ASAN_RT_NAME := asan_xros_dynamic
  endif
endif

# Note the lazy variable setting - needs to use CLANG_RT_DIR, defined in Makefile.rules
asan: LD_EXTRAS = -lclang_rt.$(ASAN_RT_NAME) -L$(CLANG_RT_DIR)
asan: SWIFTFLAGS += -sanitize=address
asan: all

libsanitizers: SWIFTFLAGS += -sanitize=address -sanitize-stable-abi
libsanitizers: all

include Makefile.rules
