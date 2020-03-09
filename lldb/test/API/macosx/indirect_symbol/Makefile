C_SOURCES := main.c
LD_EXTRAS := -L. -lindirect -lreexport

.PHONY: build-libindirect build-libreepxoprt
all: build-libindirect build-libreepxoprt a.out

include Makefile.rules

build-libindirect: indirect.c
	$(MAKE) -f $(MAKEFILE_RULES) \
		DYLIB_C_SOURCES=indirect.c DYLIB_NAME=indirect DYLIB_ONLY=YES

build-libreepxoprt: reexport.c
	$(MAKE) -f $(MAKEFILE_RULES) \
		DYLIB_C_SOURCES=reexport.c DYLIB_NAME=reexport DYLIB_ONLY=YES \
		LD_EXTRAS="-L. -lindirect -Wl,-alias_list,$(SRCDIR)/alias.list"
