ASM_SOURCES := main.s

# This is to appease Makefile.rules, there is no main.c.
C_SOURCES := main.c

ASM_OBJS := $(ASM_SOURCES:.s=.o)

LD_EXTRAS := -target arm-unknown-linux-gnueabihf \
	-fuse-ld=lld \
	-nostdlib \
	-static \
	-Wl,-e,_start

include Makefile.rules

%.o: %.s
	$(CC) $(CFLAGS) -target arm-unknown-linux-gnueabihf -c -x assembler $< -o $@