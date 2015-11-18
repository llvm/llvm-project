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
include $(LLDB_LEVEL)/../../Makefile.config

ifneq ($(HOST_OS),MingW)
ifeq (,$(findstring -DLLDB_DISABLE_PYTHON,$(CXXFLAGS)))
DO_BUILD_LLDBWrapPython = 1
BUILT_SOURCES := LLDBWrapPython.cpp
endif
endif

include $(LLDB_LEVEL)/Makefile
-include $(PROJ_OBJ_DIR)/LLDBWrapPython.cpp.d

ifeq ($(DO_BUILD_LLDBWrapPython),1)
# Drop -Wfour-char-constants,  which we are not currently clean with.
EXTRA_OPTIONS += -Wno-four-char-constants

# Drop -Wself-assign, -Wmissing-field-initializers, -Wsometimes-uninitialized,
# -Wcast-qual, and -Wdeprecated-register which we are not clean with due to SWIG
# generated cpp source.
EXTRA_OPTIONS += -Wno-missing-field-initializers -Wno-self-assign -Wno-sometimes-uninitialized -Wno-cast-qual -Wno-deprecated-register

PYTHON_DIR := $(PROJ_OBJ_ROOT)/$(BuildMode)

SWIG_SOURCES := $(shell find $(PROJ_SRC_DIR)/$(LLDB_LEVEL)/scripts -type f -name '*.swig' -print)

LLDBWrapPython.cpp lldb.py: $(PROJ_SRC_DIR)/$(LLDB_LEVEL)/scripts/Python/modify-python-lldb.py \
                            $(wildcard $(PROJ_SRC_DIR)/$(LLDB_LEVEL)/scripts/interface/*.i) \
                            ${SWIG_SOURCES}
	$(Echo) Generating LLDBWrapPython.cpp
	$(Verb) "$(PROJ_SRC_DIR)/$(LLDB_LEVEL)/scripts/prepare_bindings.py" "--src-root=$(PROJ_SRC_DIR)/$(LLDB_LEVEL)" "--target-dir=$(PROJ_OBJ_DIR)" "--config-build-dir=$(PROJ_OBJ_DIR)" "--prefix=$(PYTHON_DIR)" $(if $(DISABLE_AUTO_DEPENDENCIES),,-M) --find-swig
	$(Verb) "$(PROJ_SRC_DIR)/$(LLDB_LEVEL)/scripts/finish-swig-wrapper-classes.sh" "$(PROJ_SRC_DIR)/$(LLDB_LEVEL)" "$(PROJ_OBJ_DIR)" "$(PROJ_OBJ_DIR)" "$(PYTHON_DIR)" -m

install-local:: lldb.py
	$(Echo) Installing $(BuildMode) LLDB python modules
	$(Verb) "$(PROJ_SRC_DIR)/$(LLDB_LEVEL)/scripts/prepare_bindings.py" "--src-root=$(PROJ_SRC_DIR)/$(LLDB_LEVEL)" "--target-dir=$(PROJ_OBJ_DIR)" "--config-build-dir=$(PROJ_OBJ_DIR)" "--prefix=$(DESTDIR)$(prefix)" --find-swig

clean-local::
	$(Verb) $(RM) -f LLDBWrapPython.cpp lldb.py
endif
