##===- utils/unittest/googletest/Makefile ------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL := ../../..
include $(LEVEL)/Makefile.config
NO_MISSING_FIELD_INITIALIZERS := $(shell $(CXX) -Wno-missing-field-initializers -fsyntax-only -xc /dev/null 2>/dev/null && echo -Wno-missing-field-initializers)
NO_VARIADIC_MACROS := $(shell $(CXX) -Wno-variadic-macros -fsyntax-only -xc /dev/null 2>/dev/null && echo -Wno-variadic-macros)


LIBRARYNAME = GoogleTest
BUILD_ARCHIVE = 1
CPP.Flags += -I$(LLVM_SRC_ROOT)/utils/unittest/googletest/include/
CPP.Flags += $(NO_MISSING_FIELD_INITIALIZERS) $(NO_VARIADIC_MACROS)

ifeq ($(OS),MingW)
  CPP.Flags += -DGTEST_OS_WINDOWS=1
endif

include $(LEVEL)/Makefile.common
