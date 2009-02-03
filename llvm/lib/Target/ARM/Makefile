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
BUILT_SOURCES = ARMGenRegisterInfo.h.inc ARMGenRegisterNames.inc \
                ARMGenRegisterInfo.inc ARMGenInstrNames.inc \
                ARMGenInstrInfo.inc ARMGenAsmWriter.inc \
                ARMGenDAGISel.inc ARMGenSubtarget.inc \
                ARMGenCodeEmitter.inc

DIRS = AsmPrinter

include $(LEVEL)/Makefile.common
