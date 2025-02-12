# Build swift modules with debug info

# Don't use 'all' target.  There is a default build rule that will kick in that
# will be wrong.  We use 'first' so that the normal 'make' command (without
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

$(EXE): objc_main.m A.swift.o B.swift.o
	$(CC) $(CFLAGS) -c -I. $< -fobjc-arc -o $(BUILDDIR)/objc_main.o
	$(SWIFTC) $(patsubst -g,,$(SWIFTFLAGS)) -o $@ $(BUILDDIR)/objc_main.o $(BUILDDIR)/A.swift.o $(BUILDDIR)/B.swift.o -L$(BUILDDIR) -Xlinker -add_ast_path -Xlinker A.swiftmodule -Xlinker -add_ast_path -Xlinker B.swiftmodule
ifneq "$(CODESIGN)" ""
	$(CODESIGN) -s - "$@"
endif

%.swift.o: %.swift
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_NAME=$(patsubst %.swift.o,%,$@) \
		SWIFT_SOURCES=$(patsubst %.swift.o,%.swift,$@) \
		SWIFT_OBJC_HEADER=$(patsubst %.swift.o,%-swift.h,$@) \
		MODULENAME=$(patsubst %.swift.o,%,$@) \
		$@
