##===- utils/unittest/UnitTestMain/Makefile ----------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../../..

include $(LEVEL)/Makefile.config

LIBRARYNAME = UnitTestMain
BUILD_ARCHIVE = 1
REQUIRES_RTTI = 1

CPP.Flags += -I$(LLVM_SRC_ROOT)/utils/unittest/googletest/include
CPP.Flags += $(NO_MISSING_FIELD_INITIALIZERS) $(NO_VARIADIC_MACROS)
CPP.Flags += -DGTEST_HAS_RTTI=0
# libstdc++'s TR1 <tuple> header depends on RTTI and uses C++'0x features not
# supported by Clang, so force googletest to use its own tuple implementation.
# When we import googletest >=1.4.0, we can drop this line.
CPP.Flags += -DGTEST_HAS_TR1_TUPLE=0

include $(LEVEL)/Makefile.common
