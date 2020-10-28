MAKE_DSYM := NO
C_SOURCES := main.c
CFLAGS_EXTRAS := -Wl,-random_uuid

all: a.out b.out create-empty-corefile

create-empty-corefile:
	$(MAKE) -f $(MAKEFILE_RULES) EXE=create-empty-corefile \
		C_SOURCES=create-empty-corefile.c

b.out:
	$(MAKE)  VPATH=$(SRCDIR)/$* -I $(SRCDIR) -f $(SRCDIR)/bout.mk

include Makefile.rules
