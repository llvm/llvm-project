C_SOURCES := main.c
LD_EXTRAS = -Wl,-rpath "-Wl,$(shell pwd)" -L. -lno-nlists -lhas-nlists

include Makefile.rules

a.out: dylib_HasNlists dylib_NoNlists

dylib_HasNlists:
	"$(MAKE)" -f $(MAKEFILE_RULES) \
	DYLIB_ONLY=YES DYLIB_NAME=has-nlists DYLIB_C_SOURCES=has-nlists.c

dylib_NoNlists:
	"$(MAKE)" VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/NoNlists.mk
