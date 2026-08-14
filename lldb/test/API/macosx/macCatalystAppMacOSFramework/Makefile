C_SOURCES := main.c
LD_EXTRAS := -L. -lfoo

# Override ARCH_CFLAGS rather than adding to CFLAGS_EXTRAS: clang rejects
# -target together with the -mtargetos= the builder derives from the host
# triple. The libfoo sub-make below still picks up the plain macOS
# -mtargetos= from the command line, which is what it wants.
override ARCH_CFLAGS := -target $(ARCH)-apple-ios13.0-macabi

# FIXME: rdar://problem/54986190
override CC_TYPE=clang
override CC=xcrun clang

all: libfoo.dylib a.out

libfoo.dylib: foo.c
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES DYLIB_NAME=foo DYLIB_C_SOURCES=foo.c

include Makefile.rules
