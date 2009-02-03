##===- lib/Target/PowerPC/Makefile -------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##
LEVEL = ../../..
LIBRARYNAME = LLVMPowerPCCodeGen
TARGET = PPC

# Make sure that tblgen is run, first thing.
BUILT_SOURCES = PPCGenInstrNames.inc PPCGenRegisterNames.inc \
                PPCGenAsmWriter.inc  PPCGenCodeEmitter.inc \
                PPCGenRegisterInfo.h.inc PPCGenRegisterInfo.inc \
                PPCGenInstrInfo.inc PPCGenDAGISel.inc \
                PPCGenSubtarget.inc PPCGenCallingConv.inc

DIRS = AsmPrinter

include $(LEVEL)/Makefile.common
