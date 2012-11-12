##===- utils/TableGen/Makefile -----------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../..
TOOLNAME = llvm-tblgen
USEDLIBS = LLVMTableGen.a LLVMSupport.a

# This tool has no plugins, optimize startup time.
TOOL_NO_EXPORTS = 1

include $(LEVEL)/Makefile.common

