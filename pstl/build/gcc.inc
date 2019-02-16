#===-- gcc.inc -----------------------------------------------------------===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.TXT for details.
#
#===----------------------------------------------------------------------===##

override compiler:=g++
XHOST_FLAG = $(KEY)march=native -fno-tree-vectorize
#    XHOST_FLAG = $(KEY)mavx2 -fno-tree-vectorize
#    XHOST_FLAG = $(KEY)mavx512f -fno-tree-vectorize
 DYN_LDFLAGS += $(LINK_KEY)stdc++
# GCC 4.8.5 and early doesn't  support -fopenmp-simd; GCC 4.9 supports OpenMP 4.0 for C/C++
ifneq (, $(shell gcc -dumpversion | egrep  "^4\.9\.[0-9]"))
    CPLUS_FLAGS += -fopenmp-simd
endif
ifneq (, $(shell gcc -dumpversion | egrep  "^[5-9]\.[0-9]\.[0-9]"))
    CPLUS_FLAGS += -fopenmp-simd
#   CPLUS_FLAGS += -fdump-rtl-loop2 #use this option to enable optimization report
endif

CPLUS_FLAGS += $(FQKEY)std=$(stdver)

ifeq ($(os_name),windows)
DISABLED_WARNINGS = $(KEY)Wno-attributes #disable MinGW warnings about extended alignment
endif
