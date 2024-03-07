C_SOURCES := main.c
LD_EXTRAS := -Wl,--build-id

# For split-dwarf Debuginfod tests, we need:

# * A .DWP file (a.out.dwp)
#   Produced by Makefile.rules with MAKE_DWO and MERGE_DWOS both set to YES

# * The "full" binary: it's missing things that live in .dwo's (a.out.debug)
#   Produced by Makefile.rules with KEEP_FULL_DEBUG_BINARY set to YES and
#   SPLIT_DEBUG_SYMBOLS set to YES

# * The stripped binary (a.out)
#   Produced by Makefile.rules

# * The 'only-keep-debug' binary (a.out.dbg)
#   Produced below

# * The .uuid file (for a little easier testing code)
#   Produced here in the rule below

MAKE_DWO := YES
MERGE_DWOS := YES
SPLIT_DEBUG_SYMBOLS := YES
KEEP_FULL_DEBUG_BINARY := YES

all: a.out.uuid a.out a.out.debug a.out.dbg

a.out.dbg: a.out.debug
	$(OBJCOPY) --only-keep-debug $< $@

# Put the build-id into a file by itself for minimum effort in Python
a.out.uuid: a.out
	$(OBJCOPY) --dump-section=.note.gnu.build-id=$@ $<

include Makefile.rules
