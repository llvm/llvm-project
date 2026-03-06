SWIFT_SOURCES := main.swift

all: ImplA a.out Postprocess

include Makefile.rules
# Register a swiftmodule built of an out-of-date interface with the linker.
LD_EXTRAS = -lImpl -L$(BUILDDIR) -Xlinker -add_ast_path -Xlinker Impl.swiftmodule
SWIFTFLAGS_EXTRAS = -I$(BUILDDIR)

ImplB: ImplB.swift
	$(call ECHO,"Building an out-of-date swift interface")
	"$(MAKE)" MAKE_DSYM=YES CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		SWIFTFLAGS_EXTRAS="-I$(BUILDDIR) -enable-library-evolution" \
		VPATH=$(SRCDIR) -I $(SRCDIR) \
		DYLIB_ONLY:=YES DYLIB_NAME=Impl \
		DYLIB_SWIFT_SOURCES:=ImplB.swift \
		-f $(MAKEFILE_RULES)
	mv Impl.swiftinterface ImplB.swiftinterface
	$(call RM,Impl.swiftmodule Impl.private.swiftinterface libImpl.dylib )

ImplA: ImplB ImplA.swift
	"$(MAKE)" MAKE_DSYM=YES CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		SWIFTFLAGS_EXTRAS="-I$(BUILDDIR) -enable-library-evolution" \
		VPATH=$(SRCDIR) -I $(SRCDIR) \
		DYLIB_ONLY:=YES DYLIB_NAME=Impl \
		DYLIB_SWIFT_SOURCES:=ImplA.swift \
		-f $(MAKEFILE_RULES)
	$(call ECHO,"Moving out-of-date-swiftinterface into place")
	$(call RM_RF,libImpl.dylib.dSYM Impl.swiftmodule Impl.private.swiftinterface)
	$(call RM,ImplA.swift.o ImplB.swift.o)
	mv ImplB.swiftinterface Impl.swiftinterface

Postprocess:
	$(call ECHO,"Compiling Impl.swiftinterfact to Impl.swiftmodule")
	$(call ECHO_TO_FILE,'import Impl',trigger.swift)
	$(SWIFT_FE) -module-cache-path cache -c trigger.swift -o trigger -I.
	mv cache/Impl*.swiftmodule Impl.swiftmodule
	$(call RM,Impl.swiftinterface)
