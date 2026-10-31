ASM_SOURCES := main.s

# This is to appease Makefile.rules, there is no main.c.
C_SOURCES := main.c

ASM_OBJS := $(ASM_SOURCES:.s=.o)

# Do not automatically add the in-tree libcxx paths (we're not going to use
# the system stdlib either).
USE_SYSTEM_STDLIB := 1

LD_EXTRAS := -target arm-unknown-linux-gnueabihf \
	-fuse-ld=lld \
	-nostdlib \
	-static \
	-Wl,-e,_start

include Makefile.rules

%.o: %.s
	$(CC) $(CFLAGS) -target arm-unknown-linux-gnueabihf -c -x assembler $< -o $@
