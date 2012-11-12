##===- unittests/ExecutionEngine/MCJIT/Makefile ------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../../..
TESTNAME = MCJIT
LINK_COMPONENTS := core jit mcjit native support

include $(LEVEL)/Makefile.config
include $(LLVM_SRC_ROOT)/unittests/Makefile.unittest

# Permit these tests to use the MCJIT's symbolic lookup.
LD.Flags += $(RDYNAMIC)
