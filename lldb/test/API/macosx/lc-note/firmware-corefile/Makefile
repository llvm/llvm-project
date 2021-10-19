MAKE_DSYM := NO
C_SOURCES := main.c
#CFLAGS_EXTRAS := 

all: a.out create-empty-corefile

create-empty-corefile:
	"$(MAKE)" -f "$(MAKEFILE_RULES)" EXE=create-empty-corefile \
		C_SOURCES=create-empty-corefile.c

include Makefile.rules
