##===- lib/Target/Sparc/Makefile ---------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../../..
LIBRARYNAME = LLVMSparcCodeGen
TARGET = Sparc

# Make sure that tblgen is run, first thing.
BUILT_SOURCES = SparcGenRegisterInfo.inc SparcGenInstrInfo.inc \
		SparcGenAsmWriter.inc SparcGenDAGISel.inc \
		SparcGenSubtargetInfo.inc SparcGenCallingConv.inc

DIRS = TargetInfo MCTargetDesc

include $(LEVEL)/Makefile.common

