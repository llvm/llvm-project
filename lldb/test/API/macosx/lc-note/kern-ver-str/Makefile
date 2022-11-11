MAKE_DSYM := NO

C_SOURCES := main.c

all: a.out create-empty-corefile

create-empty-corefile:
	$(MAKE) -f $(MAKEFILE_RULES) EXE=create-empty-corefile \
	    CXX=$(CC) CXX_SOURCES=create-empty-corefile.cpp

include Makefile.rules
