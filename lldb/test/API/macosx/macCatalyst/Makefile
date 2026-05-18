C_SOURCES := main.c

USE_SYSTEM_STDLIB := 1

# FIXME: rdar://problem/54986190
# There is a Clang driver change missing on llvm.org.
override CC_TYPE=clang
override CC=xcrun clang

include Makefile.rules
