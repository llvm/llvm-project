##===- lib/ExecutionEngine/JITProfile/Makefile -------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##
LEVEL = ../../..
LIBRARYNAME = LLVMIntelJITEvents

include $(LEVEL)/Makefile.config

SOURCES := IntelJITEventListener.cpp \
  jitprofiling.c
CPPFLAGS += -I$(PROJ_OBJ_DIR)/.. -I$(PROJ_SRC_DIR)/..

include $(LLVM_SRC_ROOT)/Makefile.rules
