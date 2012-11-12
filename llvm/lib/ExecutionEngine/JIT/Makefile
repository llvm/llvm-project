##===- lib/ExecutionEngine/JIT/Makefile --------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../../..
LIBRARYNAME = LLVMJIT

# Get the $(ARCH) setting
include $(LEVEL)/Makefile.config

# Enable the X86 JIT if compiling on X86
ifeq ($(ARCH), x86)
  ENABLE_X86_JIT = 1
endif

# This flag can also be used on the command line to force inclusion
# of the X86 JIT on non-X86 hosts
ifdef ENABLE_X86_JIT
  CPPFLAGS += -DENABLE_X86_JIT
endif

# Enable the Sparc JIT if compiling on Sparc
ifeq ($(ARCH), Sparc)
  ENABLE_SPARC_JIT = 1
endif

# This flag can also be used on the command line to force inclusion
# of the Sparc JIT on non-Sparc hosts
ifdef ENABLE_SPARC_JIT
  CPPFLAGS += -DENABLE_SPARC_JIT
endif

include $(LEVEL)/Makefile.common
