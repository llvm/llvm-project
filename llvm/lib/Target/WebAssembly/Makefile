##===- lib/Target/WebAssembly/Makefile ---------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../../..
LIBRARYNAME = LLVMWebAssemblyCodeGen
TARGET = WebAssembly

# Make sure that tblgen is run, first thing.
BUILT_SOURCES = \
	WebAssemblyGenAsmWriter.inc \
	WebAssemblyGenDAGISel.inc \
	WebAssemblyGenFastISel.inc \
	WebAssemblyGenInstrInfo.inc \
	WebAssemblyGenMCCodeEmitter.inc \
	WebAssemblyGenRegisterInfo.inc \
	WebAssemblyGenSubtargetInfo.inc

DIRS = InstPrinter TargetInfo MCTargetDesc Disassembler

include $(LEVEL)/Makefile.common
