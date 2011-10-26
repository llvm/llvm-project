##===- lib/Target/Mips/Makefile ----------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../../..
LIBRARYNAME = LLVMMipsCodeGen
TARGET = Mips

# Make sure that tblgen is run, first thing.
BUILT_SOURCES = MipsGenRegisterInfo.inc MipsGenInstrInfo.inc \
                MipsGenAsmWriter.inc MipsGenCodeEmitter.inc \
                MipsGenDAGISel.inc MipsGenCallingConv.inc \
                MipsGenSubtargetInfo.inc

DIRS = InstPrinter TargetInfo MCTargetDesc

include $(LEVEL)/Makefile.common

