C_SOURCES := main.c
MAKE_DSYM := NO

include Makefile.rules

# lldb has a separate bug where this test case
# does not work if we have debug info - after
# sliding the binary, the address of `first` and
# `second` are not slid for some reason on Darwin.
main.o: main.c
	$(CC) $(CFLAGS_NO_DEBUG) -c $< -o $@

