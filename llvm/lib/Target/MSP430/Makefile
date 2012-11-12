##===- lib/Target/MSP430/Makefile --------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source 
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL = ../../..
LIBRARYNAME = LLVMMSP430CodeGen
TARGET = MSP430

# Make sure that tblgen is run, first thing.
BUILT_SOURCES = MSP430GenRegisterInfo.inc MSP430GenInstrInfo.inc \
		MSP430GenAsmWriter.inc \
		MSP430GenDAGISel.inc MSP430GenCallingConv.inc \
		MSP430GenSubtargetInfo.inc

DIRS = InstPrinter TargetInfo MCTargetDesc

include $(LEVEL)/Makefile.common

