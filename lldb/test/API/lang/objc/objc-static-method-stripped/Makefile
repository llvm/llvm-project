OBJC_SOURCES := static.m
LD_EXTRAS := -lobjc -framework Foundation

default:        a.out.stripped

a.out.stripped: a.out.dSYM
	$(STRIP) -o a.out.stripped a.out
	ln -sf a.out.dSYM a.out.stripped.dSYM

include Makefile.rules
