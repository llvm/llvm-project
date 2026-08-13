SWIFT_SOURCES=main.swift
SWIFTFLAGS_EXTRAS= -I.
LD_EXTRAS=-L.

all: clean SomeLibraryCore.swiftmodule SomeLibrary.swiftmodule a.out

include Makefile.rules

SomeLibraryCore.swiftmodule: SomeLibraryCore.swift \
                             $(call tool_prereq,$(SWIFTC_PATH)) $(call tool_prereq,$(SWIFT_FRONTEND_PATH))
	"$(MAKE)" BASENAME=$(shell basename $@ .swiftmodule) \
		SWIFTFLAGS_EXTRAS=-enable-library-evolution \
		VPATH=$(SRCDIR) -f $(SRCDIR)/../dylib.mk all

SomeLibrary.swiftmodule: SomeLibrary.swift SomeLibraryCore.swiftmodule \
                         $(call tool_prereq,$(SWIFTC_PATH)) $(call tool_prereq,$(SWIFT_FRONTEND_PATH))
	"$(MAKE)" BASENAME=$(shell basename $@ .swiftmodule) \
		SWIFTFLAGS_EXTRAS=-enable-library-evolution \
		VPATH=$(SRCDIR) LD_EXTRAS="-L$(BUILDDIR) -Xlinker -rpath -Xlinker $(BUILDDIR)" -f $(SRCDIR)/../dylib.mk all

clean::
	$(call RM_RF,*.o *.dylib *.so *.dll *.dSYM *.swiftdoc *.swiftmodule *.swiftinterface)
