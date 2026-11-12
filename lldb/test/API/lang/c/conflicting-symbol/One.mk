DYLIB_NAME := One
DYLIB_C_SOURCES := One.c OneConstant.c
DYLIB_ONLY := YES

include Makefile.rules

OneConstant.o: OneConstant.c
	$(CC) $(CFLAGS_NO_DEBUG) -c $< -o $@
