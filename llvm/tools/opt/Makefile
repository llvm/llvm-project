##===- tools/opt/Makefile ------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../..
TOOLNAME = opt
REQUIRES_EH := 1

LINK_COMPONENTS := bcreader bcwriter instrumentation scalaropts ipo \
                   transforms

include $(LEVEL)/Makefile.common
