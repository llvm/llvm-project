##===- lib/Target/IA64/Makefile -----------------------------*- Makefile -*-===##
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##
LEVEL = ../../..
LIBRARYNAME = LLVMIA64
TARGET = IA64
# Make sure that tblgen is run, first thing.
BUILT_SOURCES = IA64GenRegisterInfo.h.inc IA64GenRegisterNames.inc \
                IA64GenRegisterInfo.inc IA64GenInstrNames.inc \
                IA64GenInstrInfo.inc IA64GenAsmWriter.inc \
		IA64GenDAGISel.inc

include $(LEVEL)/Makefile.common

