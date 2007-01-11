##===- tools/bugpoint/Makefile -----------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../..

TOOLNAME = bugpoint

LINK_COMPONENTS := bcreader bcwriter asmparser instrumentation scalaropts ipo \
                   transforms linker
REQUIRES_EH := 1

include $(LEVEL)/Makefile.common
