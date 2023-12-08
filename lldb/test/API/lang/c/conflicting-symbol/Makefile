LD_EXTRAS := -L. -LOne -l$(LIB_PREFIX)One -LTwo -l$(LIB_PREFIX)Two
C_SOURCES := main.c

include Makefile.rules

.PHONY:
a.out: lib_One lib_Two

lib_%:
	$(MAKE) VPATH=$(SRCDIR)/$* -I $(SRCDIR) -f $(SRCDIR)/$*.mk
