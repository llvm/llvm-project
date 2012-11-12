##===- tools/lli/Makefile ------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL := ../..
TOOLNAME := lli

include $(LEVEL)/Makefile.config

LINK_COMPONENTS := mcjit jit interpreter nativecodegen bitreader asmparser selectiondag native

# If Intel JIT Events support is confiured, link against the LLVM Intel JIT
# Events interface library
ifeq ($(USE_INTEL_JITEVENTS), 1)
  LINK_COMPONENTS += inteljitevents
endif

# If oprofile support is confiured, link against the LLVM oprofile interface
# library
ifeq ($(USE_OPROFILE), 1)
  LINK_COMPONENTS += oprofilejit
endif

include $(LLVM_SRC_ROOT)/Makefile.rules
