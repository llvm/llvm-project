##===- tools/llvmc/driver/Makefile -------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open
# Source License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../../..
TOOLNAME = $(DRIVER_NAME)
LINK_COMPONENTS = support system
REQUIRES_EH := 1

ifneq ($(BUILTIN_PLUGINS),)
USEDLIBS = $(patsubst %,plugin_llvmc_%,$(BUILTIN_PLUGINS))
endif

include $(LEVEL)/Makefile.common
