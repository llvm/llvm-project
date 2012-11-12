##===- lib/Target/ARM/Makefile -----------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../../..
LIBRARYNAME = LLVMARMCodeGen
TARGET = ARM

# Make sure that tblgen is run, first thing.
BUILT_SOURCES = ARMGenRegisterInfo.inc ARMGenInstrInfo.inc \
		ARMGenAsmWriter.inc ARMGenAsmMatcher.inc \
                ARMGenDAGISel.inc ARMGenSubtargetInfo.inc \
                ARMGenCodeEmitter.inc ARMGenCallingConv.inc \
                ARMGenEDInfo.inc ARMGenFastISel.inc ARMGenMCCodeEmitter.inc \
                ARMGenMCPseudoLowering.inc ARMGenDisassemblerTables.inc

DIRS = InstPrinter AsmParser Disassembler TargetInfo MCTargetDesc

include $(LEVEL)/Makefile.common
