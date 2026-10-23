CXX_SOURCES := main.cpp

ifeq "$(OS)" "Windows_NT"
	ifneq "$(MAKE_PDB)" "YES"
		LD_EXTRAS += -Wl,-debug:symtab
	endif
endif

include Makefile.rules
