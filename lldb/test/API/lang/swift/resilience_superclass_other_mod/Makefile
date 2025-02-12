SWIFT_SOURCES := main.swift

all: libModWithClass.dylib libModWithSuper.dylib a.out

include Makefile.rules
LD_EXTRAS = -lModWithClass -L$(BUILDDIR)
SWIFTFLAGS_EXTRAS = -I$(BUILDDIR)

libModWithClass.dylib: ModWithClass.swift libModWithSuper.dylib
	"$(MAKE)" MAKE_DSYM=YES CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		BASENAME=ModWithClass \
		SWIFTFLAGS_EXTRAS="-I$(BUILDDIR) -enable-library-evolution" \
		LD_EXTRAS="-lModWithSuper -L$(BUILDDIR)" \
		VPATH=$(SRCDIR) -I $(SRCDIR) \
		DYLIB_ONLY:=YES DYLIB_NAME=ModWithClass \
		DYLIB_SWIFT_SOURCES:=ModWithClass.swift \
		-f $(MAKEFILE_RULES)

libModWithSuper.dylib: ModWithSuper.swift
	"$(MAKE)" MAKE_DSYM=YES CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		BASENAME=ModWithSuper \
		SWIFTFLAGS_EXTRAS="-I$(BUILDDIR) -enable-library-evolution" \
		VPATH=$(SRCDIR) -I $(SRCDIR) \
		DYLIB_ONLY:=YES DYLIB_NAME=ModWithSuper \
		DYLIB_SWIFT_SOURCES:=ModWithSuper.swift \
		-f $(MAKEFILE_RULES)
