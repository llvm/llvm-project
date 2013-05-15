##===- source/Makefile -------------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LLDB_LEVEL := ..
DIRS := API Breakpoint Commands Core DataFormatters Expression Host Interpreter Plugins Symbol Target Utility
LIBRARYNAME := lldbInitAndLog
BUILD_ARCHIVE = 1

# Although LLVM makefiles provide $(HOST_OS), we cannot use that here because it is defined by including the $(LLDB_LEVEL)/Makefile
# below. Instead, we use uname -s to detect the HOST_OS and generate LLDB_vers.c only on Mac. On Linux, the version number is
# calculated in the same way as Clang's version.
ifeq (Darwin,$(shell uname -s)) 
BUILT_SOURCES = LLDB_vers.c
endif

SOURCES := lldb-log.cpp lldb.cpp

include $(LLDB_LEVEL)/Makefile

ifeq ($(HOST_OS),Darwin)
LLDB_vers.c: $(PROJ_SRC_DIR)/$(LLDB_LEVEL)/scripts/generate-vers.pl $(PROJ_SRC_DIR)/$(LLDB_LEVEL)/lldb.xcodeproj/project.pbxproj
	"$(PROJ_SRC_DIR)/$(LLDB_LEVEL)/scripts/generate-vers.pl" "$(PROJ_SRC_DIR)/$(LLDB_LEVEL)/lldb.xcodeproj/project.pbxproj" > LLDB_vers.c
else
LLDB_REVISION := $(strip \
        $(shell $(LLVM_SRC_ROOT)/utils/GetSourceVersion $(LLVM_SRC_ROOT)/tools/lldb))

LLDB_REPOSITORY := $(strip \
        $(shell $(LLVM_SRC_ROOT)/utils/GetRepositoryPath $(LLVM_SRC_ROOT)/tools/lldb))

CPP.Defines += -DLLDB_REVISION='"$(LLDB_REVISION)"' -DLLDB_REPOSITORY='"$(LLDB_REPOSITORY)"'
endif
