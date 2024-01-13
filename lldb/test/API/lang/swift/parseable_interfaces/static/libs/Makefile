# Set the dylib variables so the rules for generating swift objects and
# modules are active.
DYLIB_NAME := $(BASENAME)
DYLIB_SWIFT_SOURCES := libs/$(BASENAME).swift

# Don't use -g for the libraries in this test.
DEBUG_INFO_FLAG :=
SWIFTFLAGS_EXTRAS := -enable-library-evolution

# Don't include the wrapped .swiftmodule on Linux to make sure we use
# the .swiftinterface. The issue here is that if the .swiftmodule is wrapped
# and linked into the library, LLDB may find it there and load it before the
# .swiftinterface loading path can be exercised.
DYLIB_HIDE_SWIFTMODULE=1

include Makefile.rules

# These are *not* Makefile.rules config variables.
ARCHIVE_NAME := lib$(BASENAME).a
ARCHIVE_OBJECTS := $(strip $(DYLIB_SWIFT_SOURCES:.swift=.swift.o))
$(ARCHIVE_NAME): $(ARCHIVE_OBJECTS) $(BASENAME).swiftmodule
	$(AR) $(ARFLAGS) $@ $<

static_only: $(ARCHIVE_NAME)

clean::
	rm -f $(BASENAME).swiftinterface
