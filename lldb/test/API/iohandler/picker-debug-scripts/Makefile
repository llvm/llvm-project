C_SOURCES := main.c
LD_EXTRAS := -L. -lalpha -lbeta -lgamma -Wl,-rpath,$(CURDIR)

a.out: libalpha libbeta libgamma

include Makefile.rules

libalpha:
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES DYLIB_NAME=alpha DYLIB_C_SOURCES=alpha.c

libbeta:
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES DYLIB_NAME=beta DYLIB_C_SOURCES=beta.c

libgamma:
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_ONLY=YES DYLIB_NAME=gamma DYLIB_C_SOURCES=gamma.c
