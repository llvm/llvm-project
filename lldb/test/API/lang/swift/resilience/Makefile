EXE=
FLAVORS := a b

# The libraries are goals in their own right: a pattern rule's output that is
# only reachable as a prerequisite counts as an intermediate file, which make
# deletes once the build is done.
all: $(FLAVORS:%=%/libmod.dylib) $(FLAVORS:%=%/main)

include Makefile.rules

# Every flavor of the library exports the same API with a different layout of S,
# and each is built in its own directory so that no artifact has to be renamed
# after linking: the paths a binary records for its serialized AST and for its
# library stay valid. A test selects a combination by copying one executable and
# one library into the top-level build directory.

%/libmod.dylib: mod.%.swift \
                $(call tool_prereq,$(SWIFTC_PATH)) $(call tool_prereq,$(SWIFT_FRONTEND_PATH))
	$(call MKDIR_P,$*)
	"$(MAKE)" -C $* MAKE_DSYM=$(MAKE_DSYM) CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		DYLIB_NAME=mod DYLIB_SWIFT_SOURCES=mod.$*.swift \
		VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/dylib.mk all
	# Withhold the binary module so that the executable, and later the
	# debugger, have to import mod from its textual interface.
	$(call RM,$*/mod.swiftmodule)

%/main: main.swift %/libmod.dylib \
        $(call tool_prereq,$(SWIFTC_PATH)) $(call tool_prereq,$(SWIFT_FRONTEND_PATH))
	"$(MAKE)" -C $* MAKE_DSYM=$(MAKE_DSYM) CC=$(CC) SWIFTC=$(SWIFTC) \
		ARCH=$(ARCH) DSYMUTIL=$(DSYMUTIL) \
		VPATH=$(SRCDIR) -I $(SRCDIR) -f $(SRCDIR)/exe.mk all

$(DSYM):
	$(call ECHO,"Already built by exe.mk")

clean::
	$(call RM_RF,$(FLAVORS) main main.dSYM main.swiftmodule libmod.dylib \
	             libmod.dylib.dSYM mod.swiftinterface mod.swiftmodule)
