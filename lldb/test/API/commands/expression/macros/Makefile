CXX_SOURCES := main.cpp

DEBUG_INFO_FLAG = -g3 -gdwarf-5

# GCC produces incorrect .debug_macro section when "-include" option is used:
# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=93075.
NO_TEST_COMMON_H := 1

include Makefile.rules
