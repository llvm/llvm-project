OBJC_SOURCES := main.m
LD_EXTRAS := -lobjc -framework Foundation

all:        a.out.stripped

a.out.stripped: a.out.dSYM
	strip -o a.out.stripped a.out

include Makefile.rules
