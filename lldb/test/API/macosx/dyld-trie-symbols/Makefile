OBJCXX_SOURCES := main.mm
EXE := a.out
MAKE_DSYM := NO
LD_EXTRAS = -dynamiclib -image_base 0x8000 -framework Foundation
CFLAGS = $(CFLAGS_NO_DEBUG)

include Makefile.rules

all: a.out a.out-stripped

a.out-stripped:
	cp a.out a.out-stripped
	strip -N a.out-stripped
