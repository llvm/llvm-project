C_SOURCES := with-debug.c without-debug.c

include Makefile.rules

without-debug.o: without-debug.c
	$(CC) $(CFLAGS_NO_DEBUG) -c $<
