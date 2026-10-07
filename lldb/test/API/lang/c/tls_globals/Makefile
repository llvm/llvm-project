C_SOURCES := main.c

DYLIB_NAME := a_dylib
DYLIB_C_SOURCES := a.c

ENABLE_THREADS := YES

ifeq "$(OS)" "Windows_NT"
	ifneq "$(MAKE_PDB)" "YES"
		# Embed a symbol table in the executable.
		LD_EXTRAS := -Wl,/debug:symtab
	endif
endif

include Makefile.rules
