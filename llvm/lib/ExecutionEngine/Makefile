##===- lib/ExecutionEngine/Makefile ------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##
LEVEL = ../..
LIBRARYNAME = LLVMExecutionEngine

include $(LEVEL)/Makefile.config

PARALLEL_DIRS = Interpreter JIT MCJIT RuntimeDyld

ifeq ($(USE_INTEL_JITEVENTS), 1)
PARALLEL_DIRS += IntelJITEvents
endif

ifeq ($(USE_OPROFILE), 1)
PARALLEL_DIRS += OProfileJIT
endif

include $(LLVM_SRC_ROOT)/Makefile.rules
