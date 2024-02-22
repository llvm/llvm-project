SWIFT_SOURCES := main.swift

all: libmod.dylib a.out

include Makefile.rules
LD_EXTRAS = -lmod -L$(BUILDDIR)
SWIFTFLAGS_EXTRAS = -I$(BUILDDIR)

libmod.dylib: mod.swift
	$(MAKE) MAKE_DSYM=YES CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		BASENAME=mod \
		SWIFTFLAGS_EXTRAS="-I$(BUILDDIR) -enable-library-evolution" \
		VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/dylib.mk all
