C_SOURCES := main.c

include Makefile.rules

all: a.out.stripped

ifeq "$(OS)" "Darwin"
STRIP_COMMAND = $(STRIP) -s keep_symbols.txt
else
STRIP_COMMAND = $(STRIP) --keep-symbol=main
endif

a.out.stripped:
	$(STRIP_COMMAND) -o a.out.stripped a.out

ifneq "$(CODESIGN)" ""
	$(CODESIGN) -fs - a.out.stripped
endif
