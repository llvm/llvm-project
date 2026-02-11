CXX_SOURCES := main.cpp

all: a.out stripped.out

include Makefile.rules

stripped.out: a.out
	$(OBJCOPY) --remove-section=.riscv.attributes a.out stripped.out
