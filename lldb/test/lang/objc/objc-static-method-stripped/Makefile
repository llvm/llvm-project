LEVEL = ../../../make

OBJC_SOURCES := static.m
LDFLAGS = $(CFLAGS) -lobjc -framework Foundation

default:        a.out.stripped

a.out.stripped: a.out.dSYM
	strip -o a.out.stripped a.out

clean::
	rm -f a.out.stripped
	rm -rf a.out.stripped.dSYM

include $(LEVEL)/Makefile.rules
