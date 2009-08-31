##===- utils/FileCheck/Makefile ----------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL = ../..
TOOLNAME = FileCheck
USEDLIBS = LLVMSupport.a LLVMSystem.a

# This tool has no plugins, optimize startup time.
TOOL_NO_EXPORTS = 1

# Don't install this utility
NO_INSTALL = 1

include $(LEVEL)/Makefile.common

