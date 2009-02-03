##===- lib/Target/Alpha/Makefile -------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##
LEVEL = ../../..
LIBRARYNAME = LLVMAlphaCodeGen
TARGET = Alpha

# Make sure that tblgen is run, first thing.
BUILT_SOURCES = AlphaGenRegisterInfo.h.inc AlphaGenRegisterNames.inc \
                AlphaGenRegisterInfo.inc AlphaGenInstrNames.inc \
                AlphaGenInstrInfo.inc AlphaGenCodeEmitter.inc \
                AlphaGenAsmWriter.inc AlphaGenDAGISel.inc \
                AlphaGenSubtarget.inc

DIRS = AsmPrinter

include $(LEVEL)/Makefile.common
