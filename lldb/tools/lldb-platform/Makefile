##===- tools/lldb-platform/Makefile ------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##
LLDB_LEVEL := ../..

TOOLNAME = lldb-platform

LLVMLibsOptions += -llldb -llldbUtility

include $(LLDB_LEVEL)/Makefile

ifeq ($(HOST_OS),Darwin)
	LLVMLibsOptions += -Wl,-rpath,@loader_path/../lib/
endif

ifeq ($(HOST_OS), $(filter $(HOST_OS), Linux FreeBSD))
	LLVMLibsOptions += -Wl,-rpath,$(LibDir)
endif
