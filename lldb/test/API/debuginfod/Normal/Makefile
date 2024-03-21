C_SOURCES := main.c

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
SAVE_FULL_DEBUG_BINARY := YES
GEN_GNU_BUILD_ID := YES

all: a.out.uuid a.out

include Makefile.rules
