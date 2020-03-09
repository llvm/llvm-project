C_SOURCES := a.c b.c
a.o: CFLAGS_EXTRAS := -gsplit-dwarf

include Makefile.rules

.PHONY: clean
clean::
	$(RM) -f a.dwo a.o b.o main
