C_SOURCES := main.c

# -Oz enables the machine outliner and -homogeneous-prolog-epilog makes it
# outline prologues deterministically.
#
# Unwind tables and debug info are off because they give lldb a 
# compiler-provided unwind plan that already describes the outlined spills.
CFLAGS_EXTRAS := -Oz -mllvm -homogeneous-prolog-epilog \
                 -fno-asynchronous-unwind-tables -fno-unwind-tables -g0

include Makefile.rules
