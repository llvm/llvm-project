##===- bindings/ocaml/llvm/Makefile ------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
# 
# This is the makefile for the Objective Caml Llvm interface.
# 
##===----------------------------------------------------------------------===##

LEVEL := ../../..
LIBRARYNAME := llvm
UsedComponents := core
UsedOcamLibs := llvm

include ../Makefile.ocaml

all-local:: copy-meta
install-local:: install-meta
uninstall-local:: uninstall-meta

DestMETA := $(PROJ_libocamldir)/META.llvm

# Easy way of generating META in the objdir
copy-meta: $(OcamlDir)/META.llvm

$(OcamlDir)/META.llvm: META.llvm
	$(Verb) $(CP) -f $< $@

install-meta:: $(OcamlDir)/META.llvm
	$(Echo) "Install $(BuildMode) $(DestMETA)"
	$(Verb) $(MKDIR) $(PROJ_libocamldir)
	$(Verb) $(DataInstall) $< "$(DestMETA)"

uninstall-meta::
	$(Echo) "Uninstalling $(DestMETA)"
	-$(Verb) $(RM) -f "$(DestMETA)"

.PHONY: copy-meta install-meta uninstall-meta
