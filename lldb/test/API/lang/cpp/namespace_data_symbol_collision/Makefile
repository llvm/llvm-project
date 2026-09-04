CXX_SOURCES := main.cpp
C_SOURCES := colliding_a.c colliding_b.c

include Makefile.rules

# The colliding_*.c files must be built WITHOUT debug info so that lldb sees
# them only as bare symtab entries (eSymbolTypeData), the same situation as
# the dyld globals that triggered the original failure.
colliding_a.o: CFLAGS = $(CFLAGS_NO_DEBUG)
colliding_b.o: CFLAGS = $(CFLAGS_NO_DEBUG)
