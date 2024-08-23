SWIFT_SOURCES := main.swift
LD_EXTRAS = -lLibrary -L$(BUILDDIR)
SWIFTFLAGS_EXTRAS = -I$(BUILDDIR)

all: libLibrary.dylib a.out

include Makefile.rules

libLibrary.dylib: Library.swift
	$(MAKE) MAKE_DSYM=YES CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		BASENAME=Library \
		SWIFTFLAGS_EXTRAS="-I$(BUILDDIR) -enable-library-evolution -emit-library -emit-module -parse-as-library -module-abi-name a" \
		VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/dylib.mk all

clean::
	$(MAKE) BASENAME=Library VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/dylib.mk clean
