SWIFT_SOURCES := main.swift

all:  libPublic.dylib a.out

include Makefile.rules
LD_EXTRAS = -lPublic -L$(BUILDDIR)
SWIFTFLAGS_EXTRAS = -I$(BUILDDIR)

libPublic.dylib: Public.swift \
                 $(call tool_prereq,$(SWIFTC_PATH)) $(call tool_prereq,$(SWIFT_FRONTEND_PATH))
	"$(MAKE)" MAKE_DSYM=YES CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		BASENAME=Public \
		SWIFTFLAGS_EXTRAS="-I$(BUILDDIR) -enable-library-evolution -enable-testing" \
		VPATH=$(SRCDIR) -I $(SRCDIR) \
		DYLIB_ONLY:=YES DYLIB_NAME=Public \
		DYLIB_SWIFT_SOURCES:=Public.swift \
		-f $(MAKEFILE_RULES)

