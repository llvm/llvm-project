CXX_SOURCES := main.cpp
USE_LIBDL := 1

a.out: lib_v hidden_lib_v other_exe

include Makefile.rules

other_exe:
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		CXX_SOURCES=other_main.cpp EXE=other.out

lib_v:
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES DYLIB_CXX_SOURCES=v.cpp DYLIB_NAME=replace_v

hidden_lib_v:
	"$(MAKE)" VPATH=$(SRCDIR)/hidden -C hidden -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES DYLIB_CXX_SOURCES=v.cpp DYLIB_NAME=replace_v
