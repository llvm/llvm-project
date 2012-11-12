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
BUILT_SOURCES = PPCGenRegisterInfo.inc \
                PPCGenAsmWriter.inc  PPCGenCodeEmitter.inc \
                PPCGenInstrInfo.inc PPCGenDAGISel.inc \
                PPCGenSubtargetInfo.inc PPCGenCallingConv.inc \
                PPCGenMCCodeEmitter.inc

DIRS = InstPrinter TargetInfo MCTargetDesc

include $(LEVEL)/Makefile.common
