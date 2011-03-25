##===- utils/llvm-lit/Makefile -----------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL = ../..

include $(LEVEL)/Makefile.common

all:: $(ToolDir)/llvm-lit

$(ToolDir)/llvm-lit: llvm-lit.in Makefile $(ToolDir)/.dir
	$(Echo) "Creating 'llvm-lit' script..."
	$(Verb)$(ECHOPATH) s=@LLVM_SOURCE_DIR@=$(LLVM_SRC_ROOT)=g > lit.tmp
	$(Verb)$(ECHOPATH) s=@LLVM_BINARY_DIR@=$(LLVM_OBJ_ROOT)=g >> lit.tmp
	$(Verb)sed -f lit.tmp $< > $@
	$(Verb)chmod +x $@
	$(Verb)rm -f lit.tmp
