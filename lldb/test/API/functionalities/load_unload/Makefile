LD_EXTRAS := -L. -lloadunload_d
CXX_SOURCES := main.cpp
USE_LIBDL := 1

a.out: lib_b lib_a lib_c lib_d hidden_lib_d

include Makefile.rules

lib_a: lib_b
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES DYLIB_CXX_SOURCES=a.cpp DYLIB_NAME=loadunload_a \
		LD_EXTRAS="-L. -lloadunload_b"

lib_b:
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES DYLIB_CXX_SOURCES=b.cpp DYLIB_NAME=loadunload_b

lib_c:
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES DYLIB_CXX_SOURCES=c.cpp DYLIB_NAME=loadunload_c

lib_d:
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES DYLIB_CXX_SOURCES=d.cpp DYLIB_NAME=loadunload_d
ifeq ($(OS),Darwin)
	install_name_tool -id @executable_path/libloadunload_d.dylib libloadunload_d.dylib
endif

hidden_lib_d: hidden
	"$(MAKE)" VPATH=$(SRCDIR)/hidden -C hidden -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES DYLIB_CXX_SOURCES=d.cpp DYLIB_NAME=loadunload_d
