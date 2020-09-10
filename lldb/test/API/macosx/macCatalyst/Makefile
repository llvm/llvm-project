C_SOURCES := main.c

CFLAGS_EXTRAS := -target $(ARCH)-apple-ios13.0-macabi

# FIXME: rdar://problem/54986190
# There is a Clang driver change missing on llvm.org.
override CC=xcrun clang

include Makefile.rules
