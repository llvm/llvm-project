SWIFT_SOURCES := main.swift

all:  libPublic.dylib a.out

include Makefile.rules
LD_EXTRAS = -lPublic -L$(BUILDDIR)
SWIFTFLAGS_EXTRAS = -I$(BUILDDIR)

libPublic.dylib: Public.swift
	"$(MAKE)" MAKE_DSYM=YES CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		BASENAME=Public \
		SWIFTFLAGS_EXTRAS="-I$(BUILDDIR) -enable-library-evolution -enable-testing" \
		VPATH=$(SRCDIR) -I $(SRCDIR) \
		DYLIB_ONLY:=YES DYLIB_NAME=Public \
		DYLIB_SWIFT_SOURCES:=Public.swift \
		-f $(MAKEFILE_RULES)

