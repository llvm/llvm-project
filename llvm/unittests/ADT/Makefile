##===- unittests/ADT/Makefile ------------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../..
TESTNAME = ADT
LINK_COMPONENTS := support

include $(LEVEL)/Makefile.config

# Xfail BitVectorTest for now on PPC Darwin.  7598360.
ifeq ($(ARCH),PowerPC)
ifeq ($(TARGET_OS),Darwin)
CPP.Flags += -DXFAIL
endif
endif

include $(LLVM_SRC_ROOT)/unittests/Makefile.unittest
