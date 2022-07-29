# Build swift modules with debug info

# Don't use 'all' target.  There is a default build rule that will kick in that
# will be wrong.  WE use 'first' so that the normal 'make' command (without
# a target) selects the first (but not 'all') target so we avoid the undesired
# default behavior.
EXE:=a.out
all: $(EXE)

SWIFT_OBJC_INTEROP=1
include Makefile.rules

# Add back the SDK settings to the swift flags.  Normally this happens
# automatically, but since we're overriding the normal swiftc invocation,
# we're not specifying SWIFT_SOURCES, and thus don't get the SDK.
SWIFTFLAGS+=-sdk "$(SWIFTSDKROOT)"

$(EXE): objc_main.m A.o B.o
	$(CC) $(CFLAGS) -c -I. $< -fobjc-arc -o $(BUILDDIR)/objc_main.o
	$(SWIFTC) $(SWIFTFLAGS) -o $@ $(BUILDDIR)/objc_main.o $(BUILDDIR)/A.o $(BUILDDIR)/B.o -L$(BUILDDIR) -Xlinker -add_ast_path -Xlinker A.swiftmodule -Xlinker -add_ast_path -Xlinker B.swiftmodule
ifneq "$(CODESIGN)" ""
	$(CODESIGN) -s - "$@"
endif

%.o: %.swift
	$(MAKE) -f $(MAKEFILE_RULES) \
		DYLIB_NAME=$(patsubst %.o,%,$@) \
		SWIFT_SOURCES=$(patsubst %.o,%.swift,$@) \
		SWIFT_OBJC_HEADER=$(patsubst %.o,%-swift.h,$@) \
		MODULENAME=$(patsubst %.o,%,$@) \
		$(patsubst %.o,%.swiftmodule,$@) \
		$(patsubst %.o,%-Swift.h,$@) \
		$@
