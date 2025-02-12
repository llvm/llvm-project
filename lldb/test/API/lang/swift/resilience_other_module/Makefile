SWIFT_SOURCES := main.swift

all: libWithDebInfo.dylib libWithoutDebInfo.dylib a.out

include Makefile.rules
LD_EXTRAS = -lWithDebInfo -lWithoutDebInfo -L$(BUILDDIR)
SWIFTFLAGS_EXTRAS = -I$(BUILDDIR)

libWithDebInfo.dylib: WithDebInfo.swift
	"$(MAKE)" MAKE_DSYM=YES CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		BASENAME=WithDebInfo \
		SWIFTFLAGS_EXTRAS="-I$(BUILDDIR) -enable-library-evolution" \
		VPATH=$(SRCDIR) -I $(SRCDIR) \
		DYLIB_ONLY:=YES DYLIB_NAME=WithDebInfo \
		DYLIB_SWIFT_SOURCES:=WithDebInfo.swift \
		-f $(MAKEFILE_RULES)

libWithoutDebInfo.dylib: WithoutDebInfo.swift
	"$(MAKE)" MAKE_DSYM=YES CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		BASENAME=WithoutDebInfo \
		SWIFTFLAGS_EXTRAS="-I$(BUILDDIR) -enable-library-evolution" \
		VPATH=$(SRCDIR) -I $(SRCDIR) \
		DYLIB_ONLY:=YES DYLIB_NAME=WithoutDebInfo \
		DYLIB_SWIFT_SOURCES:=WithoutDebInfo.swift \
		-f $(MAKEFILE_RULES)
