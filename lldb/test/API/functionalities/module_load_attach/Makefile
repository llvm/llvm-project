C_SOURCES := main.c
LD_EXTRAS := -Wl,-rpath "-Wl,$(shell pwd)"
USE_LIBDL := 1

feature:
	$(MAKE) -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES DYLIB_NAME=feature DYLIB_C_SOURCES=feature.c
all: feature

include Makefile.rules
