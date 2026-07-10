SWIFT_SOURCES=main.swift
SWIFTFLAGS_EXTRAS= -I.
LD_EXTRAS=-L.

# rpath is a Unix loader concept rejected by lld-link.
SHLIB_RPATH = -Xlinker -rpath -Xlinker $(BUILDDIR)
ifeq "$(OS)" "Windows_NT"
SHLIB_RPATH =
endif

all: clean SomeLibraryCore.swiftmodule SomeLibrary.swiftmodule a.out

include Makefile.rules

SomeLibraryCore.swiftmodule: SomeLibraryCore.swift
	"$(MAKE)" BASENAME=$(shell basename $@ .swiftmodule) \
		VPATH=$(SRCDIR) -f $(SRCDIR)/../dylib.mk all

SomeLibrary.swiftmodule: SomeLibrary.swift SomeLibraryCore.swiftmodule
	"$(MAKE)" BASENAME=$(shell basename $@ .swiftmodule) \
		VPATH=$(SRCDIR) LD_EXTRAS="-L$(BUILDDIR) $(SHLIB_RPATH)" -f $(SRCDIR)/../dylib.mk all

clean::
	$(call RM_RF,*.o *.dylib *.so *.dll *.dSYM *.swiftdoc *.swiftmodule *.swiftinterface)

