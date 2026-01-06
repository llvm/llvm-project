C_SOURCES := main.c
compiler_rt-asan: CFLAGS_EXTRAS := -fsanitize=address -g -gcolumn-info
compiler_rt-asan: all

libsanitizers-asan: CFLAGS_EXTRAS := -fsanitize=address -fsanitize-stable-abi -g -gcolumn-info
libsanitizers-asan: all

libsanitizers-traces: CFLAGS_EXTRAS := -g -gcolumn-info
libsanitizers-traces: all

include Makefile.rules
