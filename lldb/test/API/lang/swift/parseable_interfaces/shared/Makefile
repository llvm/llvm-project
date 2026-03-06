EXE := main
SWIFT_SOURCES := main.swift
LD_EXTRAS = -L$(BUILDDIR) -lAA -lBB -lCC -Xlinker -rpath -Xlinker $(BUILDDIR)
SWIFTFLAGS_EXTRAS = -I$(BUILDDIR)
SWIFT_MODULE_CACHE_FLAGS = -module-cache-path MCP

.PHONY: setup sharedA sharedB sharedC clear_modules

# setup `all` to run each of the below before building the main executable
all: setup sharedA sharedB sharedC clear_modules

include Makefile.rules

setup:
	$(call MKDIR_P,libs)
	$(call CP,$(SRCDIR)/libs/A.swift,libs/AA.swift)
	$(call CP,$(SRCDIR)/libs/B.swift,libs/BB.swift)
	$(call CP,$(SRCDIR)/libs/C.swift,libs/CC.swift)
	$(call ECHO_TO_FILE,'$(SWIFTSDKROOT)',$(BUILDDIR)/sdk-root.txt)

sharedA:
	"$(MAKE)" MAKE_DSYM=NO CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) VPATH=$(BUILDDIR) SWIFTSDKROOT=$(SWIFTSDKROOT) \
		SWIFT_MODULE_CACHE_FLAGS="$(SWIFT_MODULE_CACHE_FLAGS)" \
		BASENAME=AA -f $(SRCDIR)/libs/Makefile

sharedB:
	"$(MAKE)" MAKE_DSYM=NO CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) VPATH=$(BUILDDIR) SWIFTSDKROOT=$(SWIFTSDKROOT) \
		SWIFT_MODULE_CACHE_FLAGS="$(SWIFT_MODULE_CACHE_FLAGS)" \
		BASENAME=BB SWIFTFLAGS_EXTRAS="-I$(BUILDDIR)" LD_EXTRAS="-L$(BUILDDIR) -lAA" -f $(SRCDIR)/libs/Makefile

sharedC:
	"$(MAKE)" MAKE_DSYM=NO CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) VPATH=$(BUILDDIR) SWIFTSDKROOT=$(SWIFTSDKROOT) \
		SWIFT_MODULE_CACHE_FLAGS="$(SWIFT_MODULE_CACHE_FLAGS)" \
		BASENAME=CC -f $(SRCDIR)/libs/Makefile

clear_modules:
	$(call RM_F,*.swiftmodule)
	$(call RM_RF,libs)
	$(call RM_RF,MCP)

clean::
	"$(MAKE)" VPATH=$(BUILDDIR) SWIFTC=$(SWIFTC) BASENAME=AA -f $(SRCDIR)/libs/Makefile clean
	"$(MAKE)" VPATH=$(BUILDDIR) SWIFTC=$(SWIFTC) BASENAME=BB -f $(SRCDIR)/libs/Makefile clean
	"$(MAKE)" VPATH=$(BUILDDIR) SWIFTC=$(SWIFTC) BASENAME=CC -f $(SRCDIR)/libs/Makefile clean
	$(call RM_RF,libs)
	$(call RM_RF,MCP)
	$(call RM_F,sdk-root.txt)
