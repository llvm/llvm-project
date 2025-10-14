C_SOURCES := main.c

# For normal (non DWP) Debuginfod tests, we need:

# * The full binary: a.out.unstripped
#   Produced by Makefile.rules with SAVE_FULL_DEBUG_BINARY set to YES and
#   SPLIT_DEBUG_SYMBOLS set to YES

# * The stripped binary (a.out)
#   Produced by Makefile.rules with SPLIT_DEBUG_SYMBOLS set to YES

# * The 'only-keep-debug' binary (a.out.debug)
#   Produced below

SPLIT_DEBUG_SYMBOLS := YES
SAVE_FULL_DEBUG_BINARY := YES
GEN_GNU_BUILD_ID := YES

include Makefile.rules
