SWIFT_OBJC_INTEROP := 1
# This Makefile overwrites the default rules.
USESWIFTDRIVER := 0
EXE := a.out

all: a.out

# This test builds an Objective-C main program that imports two Swift
# .a archives.

include Makefile.rules

ifeq "$(OS)" "Darwin"
SWIFTFLAGS += -sdk "$(SWIFTSDKROOT)"
endif

a.out: main.o libFoo.a libBar.a Foo.swiftmodule Bar.swiftmodule
	$(SWIFTC) $(SWIFTFLAGS) $< -lFoo -lBar -L$(shell pwd) -o $@ \
	    -Xlinker -add_ast_path -Xlinker $(shell pwd)/Foo.swiftmodule \
	    -Xlinker -add_ast_path -Xlinker $(shell pwd)/Bar.swiftmodule
ifneq "$(CODESIGN)" ""
	$(CODESIGN) -s - "$@"
endif

main.o: main.m libFoo.a libBar.a Foo.h Bar.h \
        $(call tool_prereq,$(CC))
	$(CC) $(CFLAGS) $(MANDATORY_MODULE_BUILD_CFLAGS) -c -o $@ $< \
		-I$(shell pwd)

lib%.o: %.swift \
        $(call tool_prereq,$(SWIFTC_PATH)) $(call tool_prereq,$(SWIFT_FRONTEND_PATH))
	$(SWIFTC) $(SWIFTFLAGS) $< -c -parse-as-library -o $@ \
		-Xcc -I$(SRCDIR) \
		-Xcc -I$(SRCDIR)/objcmodules/$* \
		-module-name $*

# The module and the generated header come out of one invocation, kept separate
# from the rule above so that both are targets make can rebuild on demand.
%.swiftmodule %.h: %.swift \
        $(call tool_prereq,$(SWIFTC_PATH)) $(call tool_prereq,$(SWIFT_FRONTEND_PATH))
	$(SWIFTC) $(SWIFTFLAGS) $< -emit-module -parse-as-library \
		-emit-objc-header-path $*.h \
		-Xcc -I$(SRCDIR) \
		-Xcc -I$(SRCDIR)/objcmodules/$* \
		-module-name $*

libFoo.o Foo.swiftmodule Foo.h: \
        $(SRCDIR)/objcmodules/Foo/module.modulemap \
        $(SRCDIR)/objcmodules/Foo/CFoo.h
libBar.o Bar.swiftmodule Bar.h: \
        $(SRCDIR)/objcmodules/Bar/module.modulemap \
        $(SRCDIR)/objcmodules/Bar/CBar.h

lib%.a: lib%.o
	$(AR) $(ARFLAGS) $@ -L$(shell pwd) $<

clean::
	$(call RM_RF,*.swiftmodule *.swiftdoc *.dSYM *~ lib*.a a.out *.o)
