##===- runtime/libprofile/Makefile -------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL = ../..
BYTECODE_LIBRARY = 1
SHARED_LIBRARY = 1
LOADABLE_MODULE = 1
LIBRARYNAME = profile_rt
EXTRA_DIST = exported_symbols.lst
EXPORTED_SYMBOL_FILE = $(PROJ_SRC_DIR)/exported_symbols.lst
BYTECODE_DESTINATION = $(CFERuntimeLibDir)

include $(LEVEL)/Makefile.common
