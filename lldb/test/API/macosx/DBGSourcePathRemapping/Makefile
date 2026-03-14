BOTDIR = $(BUILDDIR)/buildbot
USERDIR = $(BUILDDIR)/user
C_SOURCES = $(BOTDIR)/main.c
LD_EXTRAS = $(BOTDIR)/relative.o

include Makefile.rules

$(EXE): relative.o
relative.o: $(BOTDIR)/relative.c
	cd $(BOTDIR) && $(CC) -c $(CFLAGS) -o $@ relative.c
