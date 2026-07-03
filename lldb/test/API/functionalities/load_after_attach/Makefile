CXX_SOURCES := main.cpp
USE_LIBDL := 1

lib_b:
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES DYLIB_CXX_SOURCES=b.cpp DYLIB_NAME=lib_b
all: lib_b

include Makefile.rules
