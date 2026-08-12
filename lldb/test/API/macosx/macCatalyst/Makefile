C_SOURCES := main.c

# The test passes the macCatalyst triple in via TRIPLE. Override ARCH_CFLAGS so
# it wins over the -mtargetos= the builder derives from the host triple, which
# clang rejects in combination with the -target below.
override ARCH_CFLAGS := -target $(TRIPLE)

USE_SYSTEM_STDLIB := 1

# FIXME: rdar://problem/54986190
# There is a Clang driver change missing on llvm.org.
override CC_TYPE=clang
override CC=xcrun clang

include Makefile.rules
