LD_EXTRAS = -L. -lto-be-removed -lpresent
C_SOURCES = main.c

include Makefile.rules

a.out: libto-be-removed libpresent

libto-be-removed: libpresent
	$(MAKE) -f $(MAKEFILE_RULES) \
	  DYLIB_ONLY=YES DYLIB_C_SOURCES=to-be-removed.c DYLIB_NAME=to-be-removed \
	  LD_EXTRAS="-L. -lpresent"

libpresent:
	$(MAKE) -f $(MAKEFILE_RULES) \
	  DYLIB_ONLY=YES DYLIB_C_SOURCES=present.c DYLIB_NAME=present 
