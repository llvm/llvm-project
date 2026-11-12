C_SOURCES := main.c

all: a.out simple misaligned

include Makefile.rules

simple:
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		C_SOURCES=simple.c EXE=simple

misaligned:
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		C_SOURCES=misaligned_first_entry.c EXE=misaligned
