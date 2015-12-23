##===- lib/IR/Makefile -------------------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##
LEVEL = ../..
LIBRARYNAME = LLVMCore
BUILD_ARCHIVE = 1

BUILT_SOURCES = $(PROJ_OBJ_ROOT)/include/llvm/IR/Intrinsics.gen \
		 $(PROJ_OBJ_ROOT)/include/llvm/IR/Attributes.inc \
		 $(PROJ_OBJ_ROOT)/lib/IR/AttributesCompatFunc.inc

include $(LEVEL)/Makefile.common

GENFILE:=$(PROJ_OBJ_ROOT)/include/llvm/IR/Intrinsics.gen
ATTRINCFILE:=$(PROJ_OBJ_ROOT)/include/llvm/IR/Attributes.inc
ATTRCOMPATFUNCINCFILE:=$(PROJ_OBJ_ROOT)/lib/IR/AttributesCompatFunc.inc

INTRINSICTD  := $(PROJ_SRC_ROOT)/include/llvm/IR/Intrinsics.td
INTRINSICTDS := $(wildcard $(PROJ_SRC_ROOT)/include/llvm/IR/Intrinsics*.td)
ATTRIBUTESTD  := $(PROJ_SRC_ROOT)/include/llvm/IR/Attributes.td
ATTRCOMPATFUNCTD  := $(PROJ_SRC_ROOT)/lib/IR/AttributesCompatFunc.td

$(ObjDir)/Intrinsics.gen.tmp: $(ObjDir)/.dir $(INTRINSICTDS) $(LLVM_TBLGEN)
	$(Echo) Building Intrinsics.gen.tmp from Intrinsics.td
	$(Verb) $(LLVMTableGen) $(call SYSPATH, $(INTRINSICTD)) -o $(call SYSPATH, $@) -gen-intrinsic

$(GENFILE): $(ObjDir)/Intrinsics.gen.tmp $(PROJ_OBJ_ROOT)/include/llvm/IR/.dir
	$(Verb) $(CMP) -s $@ $< || ( $(CP) $< $@ && \
	  $(EchoCmd) Updated Intrinsics.gen because Intrinsics.gen.tmp \
	    changed significantly. )

$(ObjDir)/Attributes.inc.tmp: $(ObjDir)/.dir $(ATTRIBUTESTD) $(LLVM_TBLGEN)
	$(Echo) Building Attributes.inc.tmp from $(ATTRIBUTESTD)
	$(Verb) $(LLVMTableGen) $(call SYSPATH, $(ATTRIBUTESTD)) -o $(call SYSPATH, $@) -gen-attrs

$(ATTRINCFILE): $(ObjDir)/Attributes.inc.tmp $(PROJ_OBJ_ROOT)/include/llvm/IR/.dir
	$(Verb) $(CMP) -s $@ $< || ( $(CP) $< $@ && \
	  $(EchoCmd) Updated Attributes.inc because Attributes.inc.tmp \
	    changed significantly. )

$(ObjDir)/AttributesCompatFunc.inc.tmp: $(ObjDir)/.dir $(ATTRCOMPATFUNCTD) $(LLVM_TBLGEN)
	$(Echo) Building AttributesCompatFunc.inc.tmp from $(ATTRCOMPATFUNCTD)
	$(Verb) $(LLVMTableGen) $(call SYSPATH, $(ATTRCOMPATFUNCTD)) -o $(call SYSPATH, $@) -gen-attrs

$(ATTRCOMPATFUNCINCFILE): $(ObjDir)/AttributesCompatFunc.inc.tmp $(PROJ_OBJ_ROOT)/include/llvm/IR/.dir
	$(Verb) $(CMP) -s $@ $< || ( $(CP) $< $@ && \
	  $(EchoCmd) Updated AttributesCompatFunc.inc because AttributesCompatFunc.inc.tmp \
	    changed significantly. )

install-local:: $(GENFILE)
	$(Echo) Installing $(DESTDIR)$(PROJ_includedir)/llvm/IR/Intrinsics.gen
	$(Verb) $(DataInstall) $(GENFILE) $(DESTDIR)$(PROJ_includedir)/llvm/IR/Intrinsics.gen

install-local:: $(ATTRINCFILE)
	$(Echo) Installing $(DESTDIR)$(PROJ_includedir)/llvm/IR/Attributes.inc
	$(Verb) $(DataInstall) $(ATTRINCFILE) $(DESTDIR)$(PROJ_includedir)/llvm/IR/Attributes.inc
