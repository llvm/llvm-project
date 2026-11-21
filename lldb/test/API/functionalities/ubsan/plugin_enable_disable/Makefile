C_SOURCES := main.c
CFLAGS_EXTRAS := -fsanitize=undefined -g

a.out: lib_dylib_with_ubsan_issue

include Makefile.rules

# Compile the directory containing the dylib into the executable so that
# dlopen() can find the dylib regardless of the working directory. This isn't
# needed on Windows which searches the executable's directory for DLLs.
ifneq "$(OS)" "Windows_NT"
CFLAGS += -DDYLIB_DIR='"$(BUILDDIR)"'
endif

lib_dylib_with_ubsan_issue:
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES DYLIB_C_SOURCES=dylib.c DYLIB_NAME=dylib_with_ubsan_issue \
		CFLAGS_EXTRAS="-fsanitize=undefined -g"
