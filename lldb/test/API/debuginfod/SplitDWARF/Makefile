C_SOURCES := main.c

# For split-dwarf Debuginfod tests, we need:

# * A .DWP file (a.out.dwp)
#   Produced by Makefile.rules with MAKE_DWP set to YES

# * The "full" binary (missing things that live in .dwo's) (a.out.unstripped)
#   Produced by Makefile.rules with SAVE_FULL_DEBUG_BINARY set to YES and
#   SPLIT_DEBUG_SYMBOLS set to YES

# * The stripped binary (a.out)
#   Produced by Makefile.rules

# * The 'only-keep-debug' binary (a.out.debug)
#   Produced below

MAKE_DWP := YES
SPLIT_DEBUG_SYMBOLS := YES
SAVE_FULL_DEBUG_BINARY := YES
GEN_GNU_BUILD_ID := YES

include Makefile.rules
