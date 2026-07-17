C_SOURCES := main.c
CFLAGS_EXTRAS := -no-pie

all: stripped.out

stripped.out : a.out
	$(STRIP) a.out -o stripped.out

include Makefile.rules
