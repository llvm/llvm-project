C_SOURCES := main.c

include Makefile.rules

all: a.out.stripped

a.out.stripped:
	$(STRIP) --keep-symbol=main -o a.out.stripped a.out

ifneq "$(CODESIGN)" ""
	$(CODESIGN) -fs - a.out.stripped
endif
