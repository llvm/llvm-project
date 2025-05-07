C_SOURCES := main.c foo.c
CFLAGS_EXTRAS := -std=c99

include Makefile.rules

a.out: main.o padding1.o padding2.o padding3.o padding4.o foo.o
	${CC} ${LDFLAGS} foo.o padding1.o padding2.o padding3.o padding4.o main.o -o a.out

%.o: $(SRCDIR)/%.s
	${CC} -c $<

#padding1.o: padding1.s
#	${CC} -c $(SRCDIR)/padding1.s

#padding2.o: padding2.s
#	${CC} -c $(SRCDIR)/padding2.s
