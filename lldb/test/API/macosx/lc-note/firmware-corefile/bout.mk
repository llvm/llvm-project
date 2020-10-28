MAKE_DSYM := NO

C_SOURCES := main.c
CFLAGS_EXTRAS := -Wl,-random_uuid

EXE := b.out

all: b.out 

include Makefile.rules
