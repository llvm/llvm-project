##===- source/Makefile -------------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LLDB_LEVEL := ..
DIRS := API Breakpoint Commands Core Expression Host Interpreter Plugins Symbol Target Utility
LIBRARYNAME := lldbInitAndLog
BUILD_ARCHIVE = 1

BUILT_SOURCES = LLDB_vers.c
SOURCES := lldb-log.cpp lldb.cpp

include $(LLDB_LEVEL)/Makefile

LLDB_vers.c: $(PROJ_SRC_DIR)/$(LLDB_LEVEL)/scripts/generate-vers.pl $(PROJ_SRC_DIR)/$(LLDB_LEVEL)/lldb.xcodeproj/project.pbxproj
	"$(PROJ_SRC_DIR)/$(LLDB_LEVEL)/scripts/generate-vers.pl" "$(PROJ_SRC_DIR)/$(LLDB_LEVEL)/lldb.xcodeproj/project.pbxproj" > LLDB_vers.c
