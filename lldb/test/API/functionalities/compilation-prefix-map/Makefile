C_SOURCES := main.c
CFLAGS_EXTRAS = -fdebug-prefix-map=$(SRCDIR)=/fake/srcdir

all: $(EXE) compilation-prefix-map.json

include Makefile.rules

compilation-prefix-map.json:
	printf '{ "/fake/srcdir": "%s" }' "$(SRCDIR)" > $(BUILDDIR)/compilation-prefix-map.json
