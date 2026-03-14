DYLIB_NAME := One
DYLIB_C_SOURCES := One.c
DYLIB_ONLY := YES
CFLAGS_EXTRAS := -O2
LD_EXTRAS := -L. -lTwo

include Makefile.rules
