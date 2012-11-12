##===- lib/Target/X86/Makefile -----------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../../..
LIBRARYNAME = LLVMX86CodeGen
TARGET = X86

# Make sure that tblgen is run, first thing.
BUILT_SOURCES = X86GenRegisterInfo.inc X86GenInstrInfo.inc \
		X86GenAsmWriter.inc X86GenAsmMatcher.inc \
                X86GenAsmWriter1.inc X86GenDAGISel.inc  \
                X86GenDisassemblerTables.inc X86GenFastISel.inc \
                X86GenCallingConv.inc X86GenSubtargetInfo.inc \
		X86GenEDInfo.inc

DIRS = InstPrinter AsmParser Disassembler TargetInfo MCTargetDesc Utils

include $(LEVEL)/Makefile.common
