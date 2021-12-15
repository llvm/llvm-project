C_SOURCES := main.c

# compile a.out and create-corefile
# create-corefile will create a custom corefile using the symbols
# addresses from the a.out binary.
all: a.out create-corefile

create-corefile:
	$(MAKE) -f $(MAKEFILE_RULES) EXE=create-corefile \
		C_SOURCES=create-corefile.c

include Makefile.rules
