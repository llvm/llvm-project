##===- tools/bugpoint/Makefile -----------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../..

TOOLNAME = bugpoint

LINK_COMPONENTS := asmparser instrumentation scalaropts ipo \
                   linker bitreader bitwriter
REQUIRES_EH := 1

include $(LEVEL)/Makefile.common
