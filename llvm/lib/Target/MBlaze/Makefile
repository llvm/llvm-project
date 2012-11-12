##===- lib/Target/MBlaze/Makefile --------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##
LEVEL = ../../..
LIBRARYNAME = LLVMMBlazeCodeGen
TARGET = MBlaze

# Make sure that tblgen is run, first thing.
BUILT_SOURCES = MBlazeGenRegisterInfo.inc MBlazeGenInstrInfo.inc \
		MBlazeGenAsmWriter.inc \
                MBlazeGenDAGISel.inc MBlazeGenAsmMatcher.inc \
                MBlazeGenCodeEmitter.inc MBlazeGenCallingConv.inc \
                MBlazeGenSubtargetInfo.inc MBlazeGenIntrinsics.inc \
                MBlazeGenEDInfo.inc

DIRS = InstPrinter AsmParser Disassembler TargetInfo MCTargetDesc

include $(LEVEL)/Makefile.common

