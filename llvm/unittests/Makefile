##===- unittests/Makefile ----------------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ..

include $(LEVEL)/Makefile.config

LIBRARYNAME = UnitTestMain
BUILD_ARCHIVE = 1
CPP.Flags += -I$(LLVM_SRC_ROOT)/utils/unittest/googletest/include/
CPP.Flags += -Wno-variadic-macros

PARALLEL_DIRS = ADT

include $(LEVEL)/Makefile.common

clean::
	$(Verb) $(RM) -f *Tests
