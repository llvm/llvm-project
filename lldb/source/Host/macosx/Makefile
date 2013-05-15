##===- source/Host/macosx/Makefile -------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LLDB_LEVEL := ../../..
LIBRARYNAME := lldbHostMacOSX
BUILD_ARCHIVE = 1

CFCPP_SOURCES = \
          $(addprefix cfcpp/,$(notdir $(wildcard $(PROJ_SRC_DIR)/cfcpp/*.cpp)))
SOURCES = $(notdir $(wildcard $(PROJ_SRC_DIR)/*.cpp $(PROJ_SRC_DIR)/*.mm)) \
          $(CFCPP_SOURCES)

include $(LLDB_LEVEL)/Makefile

CFCPP_BaseNameSources := $(sort $(basename $(CFCPP_SOURCES)))
CFCPP_OBJECTS := $(CFCPP_BaseNameSources:%=$(ObjDir)/%.o)

# Make sure the cfcpp output directory exists
$(CFCPP_OBJECTS): $(ObjDir)/cfcpp/.dir

