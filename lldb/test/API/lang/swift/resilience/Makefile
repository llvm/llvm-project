EXE=
all: libmod.a.dylib libmod.b.dylib main.a main.b

include Makefile.rules

libmod.%.dylib: mod.%.swift
	ln -sf $< mod.swift
	$(SWIFTC) $(SWIFTFLAGS) -emit-module -module-name mod mod.swift -emit-library -o $@ -Xlinker -install_name -Xlinker @executable_path/libmod.$*.dylib -###
	"$(MAKE)" MAKE_DSYM=$(MAKE_DSYM) CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		DYLIB_NAME=mod \
		VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/dylib.mk all
	mv libmod.dylib $@
	mv libmod.dylib.dSYM $@.dSYM || true
	rm mod.swiftmodule
	mv mod.swiftinterface mod.$*.swiftinterface
	mv mod.swift.o mod.$*.swift.o
	rm mod.swift

main.%: main.swift
	ln -sf libmod.$*.dylib libmod.dylib
	ln -sf libmod.$*.dylib.dSYM libmod.dylib.dSYM
	ln -sf mod.$*.swiftdoc mod.swiftdoc
	ln -sf mod.$*.swiftinterface mod.swiftinterface
	"$(MAKE)" MAKE_DSYM=$(MAKE_DSYM) CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/exe.mk all
	mv main main.$*
	mv main.dSYM main.$*.dSYM || true
	rm -rf libmod.dylib libmod.dylib.dSYM mod.swiftdoc mod.swiftmodule

$(DSYM):
	echo "Already built by exe.mk"

clean::
	rm -rf main main.a main.b *.dylib *.dSYM *.swiftdoc *.swiftmodule mod.swift *.o
