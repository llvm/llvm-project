C_SOURCES := main.c
# The memset provided by glibc may use instructions we cannot use in
# streaming mode.
CFLAGS_EXTRAS := -march=armv8-a+sve+sme+sme2 -fno-builtin-memset

include Makefile.rules
