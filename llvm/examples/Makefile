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

PARALLEL_DIRS:= BrainF Fibonacci HowToUseJIT Kaleidoscope ModuleMaker

ifeq ($(HAVE_PTHREAD),1)
PARALLEL_DIRS += ParallelJIT
endif

ifeq ($(LLVM_ON_UNIX),1)
    ifeq ($(ARCH),x86)
	PARALLEL_DIRS += ExceptionDemo
    endif
    ifeq ($(ARCH),x86_64)
	PARALLEL_DIRS += ExceptionDemo
    endif
endif

ifeq ($(filter $(BINDINGS_TO_BUILD),ocaml),ocaml)
	PARALLEL_DIRS += OCaml-Kaleidoscope
endif

include $(LEVEL)/Makefile.common
