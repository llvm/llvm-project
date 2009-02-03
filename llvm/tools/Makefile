##===- tools/Makefile --------------------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL := ..

# Build clang if present.
OPTIONAL_PARALLEL_DIRS := clang

# NOTE: The tools are organized into five groups of four consisting of one
# large and three small executables. This is done to minimize memory load
# in parallel builds.  Please retain this ordering.
PARALLEL_DIRS := llvm-config  \
                 opt llvm-as llvm-dis \
                 llc llvm-ranlib llvm-ar llvm-nm \
                 llvm-ld llvm-prof llvm-link \
		 lli gccas gccld llvm-extract llvm-db \
		 bugpoint llvm-bcanalyzer llvm-stub llvmc


include $(LEVEL)/Makefile.config

ifeq ($(ENABLE_PIC),1)
PARALLEL_DIRS += lto
endif

include $(LEVEL)/Makefile.common
