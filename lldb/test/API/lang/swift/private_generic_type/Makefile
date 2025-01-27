SWIFT_SOURCES := main.swift

all:  libPublic.dylib libPrivate.dylib a.out

include Makefile.rules
LD_EXTRAS = -lPublic -lPrivate -L$(BUILDDIR)
SWIFTFLAGS_EXTRAS = -I$(BUILDDIR)

libPrivate.dylib: Private.swift
	"$(MAKE)" MAKE_DSYM=NO CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		BASENAME=Private \
		LD_EXTRAS="-lPublic -L$(BUILDDIR)" \
		SWIFTFLAGS_EXTRAS="-I$(BUILDDIR)" \
		VPATH=$(SRCDIR) -I $(SRCDIR) \
		DYLIB_ONLY:=YES DYLIB_NAME=Private \
		DYLIB_SWIFT_SOURCES:=Private.swift \
		-f $(MAKEFILE_RULES)

libPublic.dylib: Public.swift
	"$(MAKE)" MAKE_DSYM=YES CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		BASENAME=Public \
		SWIFTFLAGS_EXTRAS="-I$(BUILDDIR)" \
		VPATH=$(SRCDIR) -I $(SRCDIR) \
		DYLIB_ONLY:=YES DYLIB_NAME=Public \
		DYLIB_SWIFT_SOURCES:=Public.swift \
		-f $(MAKEFILE_RULES)

clean::
	"$(MAKE)" BASENAME=Private VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/dylib.mk clean
	"$(MAKE)" BASENAME=Public VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/dylib.mk clean


