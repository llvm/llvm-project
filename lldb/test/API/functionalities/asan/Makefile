C_SOURCES := main.c
asan: CFLAGS_EXTRAS := -fsanitize=address -g -gcolumn-info
asan: all

libsanitizers: CFLAGS_EXTRAS := -fsanitize=address -fsanitize-stable-abi -g -gcolumn-info
libsanitizers: all

include Makefile.rules
