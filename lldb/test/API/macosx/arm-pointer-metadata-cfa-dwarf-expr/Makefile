ASM_SOURCES := main.s

# This is to appease Makefile.rules, there is no main.c
C_SOURCES := main.c

ASM_OBJS := $(ASM_SOURCES:.s=.o)

include Makefile.rules

%.o: %.s
	$(CC) $(CFLAGS) -c -x assembler $< -o $@

