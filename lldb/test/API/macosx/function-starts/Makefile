CXX_SOURCES := main.cpp
EXE := StripMe
MAKE_DSYM := NO

include Makefile.rules

main.o: main.cpp
	$(CC) $(CFLAGS_NO_DEBUG) -c $< -o $@
