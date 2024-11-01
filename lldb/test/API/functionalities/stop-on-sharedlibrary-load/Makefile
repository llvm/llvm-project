CXX_SOURCES := main.cpp
USE_LIBDL := 1

a.out: lib_a lib_b

include Makefile.rules

lib_a:
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES DYLIB_CXX_SOURCES=a.cpp DYLIB_NAME=load_a

lib_b:
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES DYLIB_CXX_SOURCES=b.cpp DYLIB_NAME=load_b


