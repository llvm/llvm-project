CXX_SOURCES := main.cpp
LD_EXTRAS := -L. -l_d -l_c -l_a -l_b

a.out: lib_b lib_a lib_c lib_d

include Makefile.rules

lib_a: lib_b
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES DYLIB_CXX_SOURCES=a.cpp DYLIB_NAME=_a \
		LD_EXTRAS="-L. -l_b"

lib_b:
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES DYLIB_CXX_SOURCES=b.cpp DYLIB_NAME=_b

lib_c:
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES DYLIB_CXX_SOURCES=c.cpp DYLIB_NAME=_c

lib_d:
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES DYLIB_CXX_SOURCES=d.cpp DYLIB_NAME=_d
