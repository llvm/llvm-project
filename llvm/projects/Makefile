##===- projects/Makefile ------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL=..

include $(LEVEL)/Makefile.config

# Compile all subdirs, except for the test suite, which lives in test-suite.
# Before 2008.06.24 it lived in llvm-test, so exclude that as well for now.
DIRS:= $(filter-out llvm-test test-suite,$(patsubst $(PROJ_SRC_DIR)/%/Makefile,%,$(wildcard $(PROJ_SRC_DIR)/*/Makefile)))

# Don't build compiler-rt, it isn't designed to be built directly.
DIRS := $(filter-out compiler-rt,$(DIRS))

# Don't build libcxx, it isn't designed to be built directly.
DIRS := $(filter-out libcxx,$(DIRS))

# Sparc cannot link shared libraries (libtool problem?)
ifeq ($(ARCH), Sparc)
DIRS := $(filter-out sample, $(DIRS))
endif

include $(PROJ_SRC_ROOT)/Makefile.rules
