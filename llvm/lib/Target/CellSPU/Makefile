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

BUILT_SOURCES = SPUGenInstrNames.inc SPUGenRegisterNames.inc \
		SPUGenAsmWriter.inc SPUGenCodeEmitter.inc \
		SPUGenRegisterInfo.h.inc SPUGenRegisterInfo.inc \
		SPUGenInstrInfo.inc SPUGenDAGISel.inc \
		SPUGenSubtarget.inc SPUGenCallingConv.inc

DIRS = AsmPrinter

include $(LEVEL)/Makefile.common
