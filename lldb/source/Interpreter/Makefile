##===- source/Interpreter/Makefile  ------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LLDB_LEVEL := ../..
LIBRARYNAME := lldbInterpreter
BUILD_ARCHIVE = 1

BUILT_SOURCES := LLDBWrapPython.cpp

include $(LLDB_LEVEL)/Makefile
-include $(PROJ_OBJ_DIR)/LLDBWrapPython.cpp.d

# edit-swig-python-wrapper-file.py needs $(SRCROOT)
export SRCROOT := $(PROJ_SRC_DIR)/$(LLDB_LEVEL)

PYTHON_DIR := $(PROJ_OBJ_ROOT)/$(BuildMode)

LLDBWrapPython.cpp lldb.py: $(PROJ_SRC_DIR)/$(LLDB_LEVEL)/scripts/Python/modify-python-lldb.py \
                            $(PROJ_SRC_DIR)/$(LLDB_LEVEL)/scripts/Python/edit-swig-python-wrapper-file.py \
                            $(wildcard $(PROJ_SRC_DIR)/$(LLDB_LEVEL)/scripts/Python/interface/*.i)
	$(Echo) Generating LLDBWrapPython.cpp
	$(Verb) "$(PROJ_SRC_DIR)/$(LLDB_LEVEL)/scripts/build-swig-wrapper-classes.sh" "$(PROJ_SRC_DIR)/$(LLDB_LEVEL)" "$(PROJ_OBJ_DIR)" "$(PROJ_OBJ_DIR)" "$(PYTHON_DIR)" -m $(if $(DISABLE_AUTO_DEPENDENCIES),,-M)
	$(Verb) "$(PROJ_SRC_DIR)/$(LLDB_LEVEL)/scripts/finish-swig-wrapper-classes.sh" "$(PROJ_SRC_DIR)/$(LLDB_LEVEL)" "$(PROJ_OBJ_DIR)" "$(PROJ_OBJ_DIR)" "$(PYTHON_DIR)" -m

install-local:: lldb.py
	$(Echo) Installing $(BuildMode) LLDB python modules
	$(Verb) "$(PROJ_SRC_DIR)/$(LLDB_LEVEL)/scripts/finish-swig-wrapper-classes.sh" "$(PROJ_SRC_DIR)/$(LLDB_LEVEL)" "$(PROJ_OBJ_DIR)" "$(PROJ_OBJ_DIR)" "" -m

clean-local::
	$(Verb) $(RM) -f LLDBWrapPython.cpp lldb.py
