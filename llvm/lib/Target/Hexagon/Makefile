##===- lib/Target/Hexagon/Makefile -------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##
LEVEL = ../../..
LIBRARYNAME = LLVMHexagonCodeGen
TARGET = Hexagon

# Make sure that tblgen is run, first thing.
BUILT_SOURCES = HexagonGenRegisterInfo.inc \
                HexagonGenInstrInfo.inc  \
                HexagonGenAsmWriter.inc \
                HexagonGenDAGISel.inc HexagonGenSubtargetInfo.inc \
                HexagonGenCallingConv.inc \
                HexagonGenDFAPacketizer.inc

DIRS = InstPrinter TargetInfo MCTargetDesc

include $(LEVEL)/Makefile.common
