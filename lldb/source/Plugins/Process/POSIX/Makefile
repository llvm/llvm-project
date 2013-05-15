##===- source/Plugins/Process/POSIX/Makefile ---------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LLDB_LEVEL := ../../../..
LIBRARYNAME := lldbPluginProcessPOSIX
BUILD_ARCHIVE = 1

include $(LLDB_LEVEL)/../../Makefile.config

# Extend the include path so we may locate UnwindLLDB.h
CPPFLAGS += -I$(PROJ_SRC_DIR)/$(LLDB_LEVEL)/source/Plugins/Utility

ifeq ($(HOST_OS),Linux)
CPPFLAGS += -I$(PROJ_SRC_DIR)/$(LLDB_LEVEL)/source/Plugins/Process/Linux

# Disable warning for now as offsetof is used with an index into a structure member array
# in defining register info tables.
CPPFLAGS += -Wno-extended-offsetof
endif

ifeq ($(HOST_OS),FreeBSD)
# Extend the include path so we may locate ProcessMonitor
CPPFLAGS += -I$(PROJ_SRC_DIR)/$(LLDB_LEVEL)/source/Plugins/Process/FreeBSD
endif

include $(LLDB_LEVEL)/Makefile
