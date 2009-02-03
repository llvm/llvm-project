##===- lib/Target/Mips/Makefile ----------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##
LEVEL = ../../..
LIBRARYNAME = LLVMMips
TARGET = Mips

# Make sure that tblgen is run, first thing.
BUILT_SOURCES = MipsGenRegisterInfo.h.inc MipsGenRegisterNames.inc \
                MipsGenRegisterInfo.inc MipsGenInstrNames.inc \
                MipsGenInstrInfo.inc MipsGenAsmWriter.inc \
                MipsGenDAGISel.inc MipsGenCallingConv.inc \
                MipsGenSubtarget.inc

include $(LEVEL)/Makefile.common

