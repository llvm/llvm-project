C_SOURCES := main.c
LD_EXTRAS := -L. -Wl,-lazy_library -Wl,libmy_library.dylib -lmy_library

.PHONY: build-lazylib
all: build-lazylib a.out

include Makefile.rules

a.out : main.c

build-lazylib: library.c
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_C_SOURCES=library.c DYLIB_NAME=my_library DYLIB_ONLY=YES \
		LD_EXTRAS=""

