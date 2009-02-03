##===- lib/Target/PIC16/Makefile ---------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source 
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../../..
LIBRARYNAME = LLVMPIC16
TARGET = PIC16

# Make sure that tblgen is run, first thing.
BUILT_SOURCES = PIC16GenRegisterInfo.h.inc PIC16GenRegisterNames.inc \
		PIC16GenRegisterInfo.inc PIC16GenInstrNames.inc \
		PIC16GenInstrInfo.inc PIC16GenAsmWriter.inc \
		PIC16GenDAGISel.inc PIC16GenCallingConv.inc \
		PIC16GenSubtarget.inc

include $(LEVEL)/Makefile.common

