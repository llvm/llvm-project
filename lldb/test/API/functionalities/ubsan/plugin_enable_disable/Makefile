C_SOURCES := main.c
CFLAGS_EXTRAS := -fsanitize=undefined -g

a.out: lib_dylib_with_ubsan_issue

include Makefile.rules

lib_dylib_with_ubsan_issue:
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES DYLIB_C_SOURCES=dylib.c DYLIB_NAME=dylib_with_ubsan_issue \
		CFLAGS_EXTRAS="-fsanitize=undefined -g"
