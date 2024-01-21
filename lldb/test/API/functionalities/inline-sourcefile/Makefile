C_SOURCES := main.c
CFLAGS_EXTRAS := -gdwarf-5

include Makefile.rules

OBJECTS += inline.o

$(EXE): main.c inline.o

%.o: %.ll
	$(CC) $< -c -o $@
