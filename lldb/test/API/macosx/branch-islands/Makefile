C_SOURCES := main.c
CFLAGS_EXTRAS := -std=c99
# Link foo.o after the padding so that main's call to foo needs a branch island.
LD_EXTRAS := padding1.o padding2.o padding3.o padding4.o foo.o

include Makefile.rules

a.out: $(LD_EXTRAS)

foo.o: foo.c
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.s
	$(CC) $(CFLAGS) -c -o $@ $<
