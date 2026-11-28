C_SOURCES := main.c weak.c
include Makefile.rules

no-debug-strong.o: CFLAGS += -g0
symbol-only.o: CFLAGS += -g0
shared-strong.o: CFLAGS += -g0

localized.o: localized-input.c
	$(CC) $(CFLAGS) -c $< -o $@
	$(OBJCOPY) --localize-symbol=action $@
