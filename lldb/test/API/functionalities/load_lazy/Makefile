CXX_SOURCES := main.cpp
USE_LIBDL := 1

all: t2_0 t2_1 t1 a.out

include Makefile.rules

ifeq "$(OS)" "Darwin"
	# In macOS 12, dyld switched to using chained fixups. As a result, all
	# symbols are bound at launch and there are no lazy pointers any more.
	# Since we wish to import/dlopen() a dylib with missing symbols, we need
	# to use a weak import. This applies to all macOS 12-aligned OS releases,
	# e.g. iOS 15, etc.
	LINKFLAGS := "-Wl,-weak-lt2_0"
else
	LINKFLAGS := "-lt2_0"
endif

t1: t2_0
	$(MAKE) VPATH=$(SRCDIR) -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES DYLIB_C_SOURCES=t1.c DYLIB_NAME=t1 LD_EXTRAS="-L. $(LINKFLAGS)"

t2_0:
	$(MAKE) VPATH=$(SRCDIR) -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES DYLIB_C_SOURCES=t2_0.c DYLIB_NAME=t2_0

t2_1:
	$(MAKE) VPATH=$(SRCDIR) -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES DYLIB_C_SOURCES=t2_1.c DYLIB_NAME=t2_1
