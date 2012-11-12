##===- lib/ExecutionEngine/OProfileJIT/Makefile ------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##
LEVEL = ../../..
LIBRARYNAME = LLVMOProfileJIT

include $(LEVEL)/Makefile.config

SOURCES += OProfileJITEventListener.cpp \
  OProfileWrapper.cpp
CPPFLAGS += -I$(PROJ_OBJ_DIR)/.. -I$(PROJ_SRC_DIR)/..

include $(LLVM_SRC_ROOT)/Makefile.rules
