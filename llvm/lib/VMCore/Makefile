##===- lib/VMCore/Makefile ---------------------------------*- Makefile -*-===##
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

BUILT_SOURCES = $(PROJ_OBJ_ROOT)/include/llvm/Intrinsics.gen

include $(LEVEL)/Makefile.common

GENFILE:=$(PROJ_OBJ_ROOT)/include/llvm/Intrinsics.gen

INTRINSICTD  := $(PROJ_SRC_ROOT)/include/llvm/Intrinsics.td
INTRINSICTDS := $(wildcard $(PROJ_SRC_ROOT)/include/llvm/Intrinsics*.td)

$(ObjDir)/Intrinsics.gen.tmp: $(ObjDir)/.dir $(INTRINSICTDS) $(LLVM_TBLGEN)
	$(Echo) Building Intrinsics.gen.tmp from Intrinsics.td
	$(Verb) $(LLVMTableGen) $(call SYSPATH, $(INTRINSICTD)) -o $(call SYSPATH, $@) -gen-intrinsic

$(GENFILE): $(ObjDir)/Intrinsics.gen.tmp
	$(Verb) $(CMP) -s $@ $< || ( $(CP) $< $@ && \
	  $(EchoCmd) Updated Intrinsics.gen because Intrinsics.gen.tmp \
	    changed significantly. )

install-local:: $(GENFILE)
	$(Echo) Installing $(DESTDIR)$(PROJ_includedir)/llvm/Intrinsics.gen
	$(Verb) $(DataInstall) $(GENFILE) $(DESTDIR)$(PROJ_includedir)/llvm/Intrinsics.gen
