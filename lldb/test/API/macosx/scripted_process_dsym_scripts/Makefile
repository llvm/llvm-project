DYLIB_ONLY := YES
DYLIB_NAME := payload
DYLIB_C_SOURCES := payload.c
MAKE_DSYM := YES
# Package the script in the dSYM, where LLDB looks for libpayload.dylib's.
DSFLAGS_EXTRAS = --embed-resource $(SRCDIR)/payload.py=Python/libpayload.py

include Makefile.rules
