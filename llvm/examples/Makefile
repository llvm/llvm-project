##===- examples/Makefile -----------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL=..

include $(LEVEL)/Makefile.config

PARALLEL_DIRS:= BrainF Fibonacci HowToUseJIT ModuleMaker

ifeq ($(HAVE_PTHREAD),1)
PARALLEL_DIRS += ParallelJIT 
endif

include $(LEVEL)/Makefile.common
