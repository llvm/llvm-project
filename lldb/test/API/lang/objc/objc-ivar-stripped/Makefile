OBJC_SOURCES := main.m
LD_EXTRAS := -lobjc -framework Foundation

all:        a.out.stripped

include Makefile.rules

a.out.stripped: a.out.dSYM
	$(STRIP) -o a.out.stripped a.out
ifneq "$(CODESIGN)" ""
	$(CODESIGN) -fs - a.out.stripped
endif
