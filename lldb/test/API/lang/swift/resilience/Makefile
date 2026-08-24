EXE=
all: libmod.a.dylib libmod.b.dylib main.a main.b

include Makefile.rules

# Both recipes below build into flavor-independent names (libmod.dylib,
# mod.swiftinterface, main, ...) and rename the results per flavor afterwards,
# so they are only correct when the directory does not already contain the
# results of an earlier build or test run: `mv` descends into an existing
# destination directory rather than replacing it, and writing to a leftover
# symlink follows it instead of overwriting it. Clear both before building.

libmod.%.dylib: mod.%.swift \
                $(call tool_prereq,$(SWIFTC_PATH)) $(call tool_prereq,$(SWIFT_FRONTEND_PATH))
	$(call RM_RF,$@.dSYM libmod.dylib libmod.dylib.dSYM \
	             mod.swiftinterface mod.$*.swiftinterface mod.swiftmodule)
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

# The tool dependencies belong on the concrete targets: main.% also matches
# main.swift, and a pattern rule whose prerequisite is its own target is only
# harmless as long as it has no prerequisite newer than that target.
main.a main.b: $(call tool_prereq,$(SWIFTC_PATH)) $(call tool_prereq,$(SWIFT_FRONTEND_PATH))

main.%: main.swift libmod.%.dylib
	$(call RM_RF,main.$*.dSYM main main.dSYM)
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
