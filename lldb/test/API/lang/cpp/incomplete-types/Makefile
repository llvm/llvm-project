CXX_SOURCES := length.cpp a.o main.o
EXE := nolimit

all: limit nolimit

include Makefile.rules

# Force a.cpp to be built with no debug information
a.o: CFLAGS = $(CFLAGS_NO_DEBUG)

# The default testsuite setup forces -fno-limit-debug-info. Let's not rely on
# CFLAGS_EXTRAS being passed after the default arguments. This rule makes
# sure the variable used by Makefile.rules for this argument is cleared.
main.o: NO_LIMIT_DEBUG_INFO_FLAGS = ""
main.o: CFLAGS_EXTRAS = -flimit-debug-info

limit: a.o main.o
	mkdir -p build_limit
	"$(MAKE)" -C $(BUILDDIR)/build_limit -f $(MAKEFILE_RULES) \
		EXE=../limit CXX_SOURCES="length.cpp ../a.o ../main.o" \
		CFLAGS_EXTRAS=-flimit-debug-info NO_LIMIT_DEBUG_INFO_FLAGS=""

