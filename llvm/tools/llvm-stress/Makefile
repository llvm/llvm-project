##===- tools/llvm-stress/Makefile --------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL := ../..
TOOLNAME := llvm-stress
LINK_COMPONENTS := object
LINK_COMPONENTS := bitreader bitwriter asmparser instrumentation scalaropts ipo

# This tool has no plugins, optimize startup time.
TOOL_NO_EXPORTS = 1

include $(LEVEL)/Makefile.common
