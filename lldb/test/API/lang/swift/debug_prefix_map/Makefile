SWIFT_SOURCES=srcroot/main.swift

# A common use case for -debug-prefix-map is to invoke the compiler such
# that its working directory is the root directory of your source tree, and
# re-map the absolute path of that location to ".". Then, a user on a
# different machine can debug the binary by invoking lldb in their source
# root (which could be at a completely different absolute path) and it will
# be able to find the source files.
SWIFTFLAGS_EXTRAS = -debug-prefix-map $(BUILDDIR)=.

include Makefile.rules

clean::
	rm -rf *.swiftmodule *.swiftdoc *.dSYM *~ a.out *.o
