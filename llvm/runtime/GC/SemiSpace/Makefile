##===- runtime/GC/SemiSpace/Makefile -----------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL = ../../..
BYTECODE_LIBRARY = 1
LIBRARYNAME = gcsemispace
BYTECODE_DESTINATION = $(CFERuntimeLibDir)
EXPORTED_SYMBOL_FILE = $(PROJ_SRC_DIR)/../gc_exported_symbols.lst

include $(LEVEL)/Makefile.common

CompileCommonOpts := $(filter-out -pedantic,$(CompileCommonOpts))
CompileCommonOpts := $(filter-out -Wno-long-long,$(CompileCommonOpts))
