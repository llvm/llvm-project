EXE=
all: libmod.a.dylib libmod.b.dylib main.a main.b

include Makefile.rules

libmod.%.dylib: mod.%.swift
	$(call LN_SF,$<,mod.swift)
	$(SWIFTC) $(SWIFTFLAGS) -emit-module -module-name mod mod.swift -emit-library -o $@ -Xlinker -install_name -Xlinker @executable_path/libmod.$*.dylib -###
	"$(MAKE)" MAKE_DSYM=$(MAKE_DSYM) CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		DYLIB_NAME=mod \
		VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/dylib.mk all
	mv libmod.dylib $@
	mv libmod.dylib.dSYM $@.dSYM || true
	$(call RM,mod.swiftmodule)
	mv mod.swiftinterface mod.$*.swiftinterface
	mv mod.swift.o mod.$*.swift.o
	$(call RM,mod.swift)

main.%: main.swift
	$(call LN_SF,libmod.$*.dylib,libmod.dylib)
	$(call LN_SF,libmod.$*.dylib.dSYM,libmod.dylib.dSYM)
	$(call LN_SF,mod.$*.swiftdoc,mod.swiftdoc)
	$(call LN_SF,mod.$*.swiftinterface,mod.swiftinterface)
	"$(MAKE)" MAKE_DSYM=$(MAKE_DSYM) CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/exe.mk all
	mv main main.$*
	mv main.dSYM main.$*.dSYM || true
	$(call RM_RF,libmod.dylib libmod.dylib.dSYM mod.swiftdoc mod.swiftmodule)

$(DSYM):
	$(call ECHO,"Already built by exe.mk")

clean::
	$(call RM_RF,main main.a main.b *.dylib *.dSYM *.swiftdoc *.swiftmodule mod.swift *.o)
