C_SOURCES := main.c
LD_EXTRAS := -Wl,--build-id

# For normal (non DWP) Debuginfod tests, we need:

# * The "full" binary: a.out.debug
#   Produced by Makefile.rules with KEEP_FULL_DEBUG_BINARY set to YES and
#   SPLIT_DEBUG_SYMBOLS set to YES

# * The stripped binary (a.out)
#   Produced by Makefile.rules with SPLIT_DEBUG_SYMBOLS set to YES

# * The 'only-keep-debug' binary (a.out.dbg)
#   Produced below

# * The .uuid file (for a little easier testing code)
#   Produced below

# Don't strip the debug info from a.out:
SPLIT_DEBUG_SYMBOLS := YES
KEEP_FULL_DEBUG_BINARY := YES

all: a.out a.out.uuid a.out.dbg a.out.debug

# Put the build-id into a file by itself for minimum effort in Python
a.out.uuid: a.out
	$(OBJCOPY) --dump-section=.note.gnu.build-id=$@ $<

a.out.dbg: a.out.debug
	$(OBJCOPY) --only-keep-debug $< $@

include Makefile.rules
