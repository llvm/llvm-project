##===- lib/Target/CellSPU/Makefile -------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../../..
LIBRARYNAME = LLVMCellSPUCodeGen
TARGET = SPU
BUILT_SOURCES = SPUGenInstrInfo.inc SPUGenRegisterInfo.inc \
		SPUGenAsmWriter.inc SPUGenCodeEmitter.inc \
		SPUGenDAGISel.inc \
		SPUGenSubtargetInfo.inc SPUGenCallingConv.inc

DIRS = TargetInfo MCTargetDesc

include $(LEVEL)/Makefile.common
