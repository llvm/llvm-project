#===- lib/Target/Makefile ----------------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../..
LIBRARYNAME = LLVMTarget
BUILD_ARCHIVE = 1

# We include this early so we can access the value of TARGETS_TO_BUILD as the
# value for PARALLEL_DIRS which must be set before Makefile.rules is included
include $(LEVEL)/Makefile.config

PARALLEL_DIRS := $(TARGETS_TO_BUILD)

include $(LLVM_SRC_ROOT)/Makefile.rules
