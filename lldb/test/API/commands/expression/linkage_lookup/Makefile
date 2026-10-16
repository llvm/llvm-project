CXX_SOURCES := main.cpp decoy.cpp
DYLIB_NAME := LinkageProbe
DYLIB_CXX_SOURCES := probe.cpp
LD_EXTRAS := -Wl,-U,_missing_weak

include Makefile.rules

# Keep the decoys in the symbol table without adding DWARF functions. This
# isolates the JIT's symbol-name lookup from the debug-info function lookup.
decoy.o: CXXFLAGS += -g0
