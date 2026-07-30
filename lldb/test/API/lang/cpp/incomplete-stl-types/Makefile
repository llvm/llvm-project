CXX_SOURCES := main.cpp f.cpp

include Makefile.rules

# Force main.cpp to be built with no debug information
main.o: CFLAGS = $(CFLAGS_NO_DEBUG)

# And force -flimit-debug-info on the rest.
f.o: CFLAGS_EXTRAS += $(LIMIT_DEBUG_INFO_FLAGS)
