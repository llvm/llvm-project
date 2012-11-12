##===- lib/Target/NVPTX/Makefile ---------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../../..
LIBRARYNAME = LLVMNVPTXCodeGen
TARGET = NVPTX

# Make sure that tblgen is run, first thing.
BUILT_SOURCES = NVPTXGenAsmWriter.inc \
		NVPTXGenDAGISel.inc \
		NVPTXGenInstrInfo.inc \
		NVPTXGenRegisterInfo.inc \
		NVPTXGenSubtargetInfo.inc

DIRS = InstPrinter TargetInfo MCTargetDesc

include $(LEVEL)/Makefile.common
