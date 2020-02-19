DYLIB_NAME := Two
DYLIB_C_SOURCES := Two.c
DYLIB_ONLY := YES
CFLAGS_EXTRAS := -g -O2 -glldb

include Makefile.rules
