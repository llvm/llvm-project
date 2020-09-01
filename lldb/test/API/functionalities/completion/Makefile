CXX_SOURCES := main.cpp
USE_LIBDL := 1

a.out: lib_shared

lib_shared:
	$(MAKE) -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES DYLIB_CXX_SOURCES=shared.cpp DYLIB_NAME=shared

include Makefile.rules
