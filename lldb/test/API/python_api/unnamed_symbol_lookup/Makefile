C_SOURCES := main.c

include Makefile.rules

all: a.out.stripped

ifeq "$(OS)" "Darwin"
STRIP_COMMAND = $(STRIP) -s keep_symbols.txt
else
STRIP_COMMAND = $(STRIP) --keep-symbol=main
endif

a.out.stripped: a.out
ifeq "$(OS)" "Darwin"
	echo "_main" > keep_symbols.txt
	$(STRIP) -s keep_symbols.txt -o a.out.stripped a.out
else
	$(STRIP) --keep-symbol=main -o a.out.stripped a.out
endif

ifneq "$(CODESIGN)" ""
	$(CODESIGN) -fs - a.out.stripped
endif
