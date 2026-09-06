C_SOURCES := main.c

ifeq "$(OS)" "Windows_NT"
	# Embed a symbol table in the executable.
	LD_EXTRAS := -Wl,/debug:symtab
endif

include Makefile.rules
