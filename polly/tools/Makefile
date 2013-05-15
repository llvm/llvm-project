##===- tools/Makefile --------------------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL := ..
DIRS :=

include $(LEVEL)/Makefile.config

# GPU Runtime Support
ifeq ($(CUDALIB_FOUND), yes)
 DIRS += GPURuntime
endif

include $(LEVEL)/Makefile.common
