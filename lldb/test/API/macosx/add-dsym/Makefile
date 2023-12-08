C_SOURCES = main.c

include Makefile.rules

all: a.out.dSYM hide.app/Contents/a.out.dSYM

hide.app/Contents/a.out.dSYM:
	mkdir hide.app
	mkdir hide.app/Contents
	mv a.out.dSYM hide.app/Contents
	strip -x a.out
ifneq "$(CODESIGN)" ""
	$(CODESIGN) -fs - a.out
endif
