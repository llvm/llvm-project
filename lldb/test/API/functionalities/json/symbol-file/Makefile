C_SOURCES := main.c

all: stripped.out

stripped.out : a.out
ifeq "$(OS)" "Darwin"
	strip a.out -o stripped.out
else
	$(STRIP) a.out -o stripped.out
endif

include Makefile.rules
