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

LIBRARYNAME = gtest_main
BUILD_ARCHIVE = 1
REQUIRES_RTTI = 1

CPP.Flags += -I$(LLVM_SRC_ROOT)/utils/unittest/googletest/include
CPP.Flags += $(NO_MISSING_FIELD_INITIALIZERS) $(NO_VARIADIC_MACROS)
CPP.Flags += -DGTEST_HAS_RTTI=0
# libstdc++'s TR1 <tuple> header depends on RTTI and uses C++'0x features not
# supported by Clang, so force googletest to use its own tuple implementation.
CPP.Flags += -DGTEST_USE_OWN_TR1_TUPLE

# Disable pthreads if LLVM was configured without them.
ifneq ($(HAVE_PTHREAD), 1)
  CPP.Flags += -DGTEST_HAS_PTHREAD=0
endif

NO_INSTALL = 1

include $(LEVEL)/Makefile.common
