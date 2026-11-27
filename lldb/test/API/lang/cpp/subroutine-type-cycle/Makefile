include Makefile.rules

OBJECTS += cycle.o

$(EXE): cycle.o

%.o: %.ll
	$(CC) $(CFLAGS) $< -c -o $@
