C_SOURCES := main.c

override TRIPLE := $(ARCH)-apple-ios13.1-macabi
CFLAGS_EXTRAS := -target $(TRIPLE)

USE_SYSTEM_STDLIB := 1

# FIXME: rdar://problem/54986190
# There is a Clang driver change missing on llvm.org.
override CC=xcrun clang

include Makefile.rules
