MAKE_DSYM := NO
C_SOURCES := main.c
LD_EXTRAS := -L. -lone -ltwo

.PHONY: libone.dylib libtwo.dylib
all: libone.dylib libtwo.dylib a.out create-empty-corefile 

create-empty-corefile:
	"$(MAKE)" -f "$(MAKEFILE_RULES)" EXE=create-multibin-corefile \
		CXX_SOURCES=create-multibin-corefile.cpp

libone.dylib: one.c
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES DYLIB_NAME=one DYLIB_C_SOURCES=one.c

libtwo.dylib: two.c
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES DYLIB_NAME=two DYLIB_C_SOURCES=two.c

include Makefile.rules
