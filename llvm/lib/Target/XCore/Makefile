##===- lib/Target/XCore/Makefile ---------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../../..
LIBRARYNAME = LLVMXCoreCodeGen
TARGET = XCore

# Make sure that tblgen is run, first thing.
BUILT_SOURCES = XCoreGenRegisterInfo.inc XCoreGenInstrInfo.inc \
		XCoreGenAsmWriter.inc \
                XCoreGenDAGISel.inc XCoreGenCallingConv.inc \
		XCoreGenSubtargetInfo.inc

DIRS = TargetInfo MCTargetDesc

include $(LEVEL)/Makefile.common

