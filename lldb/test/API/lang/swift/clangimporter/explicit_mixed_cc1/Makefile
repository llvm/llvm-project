SWIFT_SOURCES := main.swift
C_SOURCES := c.c
SWIFT_ENABLE_EXPLICIT_MODULES := YES
SWIFTFLAGS_EXTRAS = -I$(SRCDIR) -I.  -Xfrontend -experimental-clang-importer-direct-cc1-scan
LD_EXTRAS = B.swift.o -Xlinker -add_ast_path -Xlinker $(BUILDDIR)/B.swiftmodule

all: B.swift.o $(EXE)

include Makefile.rules

B.swift.o: B.swift
	"$(MAKE)" MAKE_DSYM=$(MAKE_DSYM) CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		SWIFT_SOURCES=B.swift \
                MODULENAME=B \
		SWIFT_ENABLE_EXPLICIT_MODULES=NO \
                SWIFTFLAGS_EXTRAS="-I$(SRCDIR) -parse-as-library" \
		VPATH=$(SRCDIR) -I $(SRCDIR) -f $(MAKEFILE_RULES) \
                B.swift.o
