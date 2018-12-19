#===-- clang.inc ---------------------------------------------------------===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.TXT for details.
#
#===----------------------------------------------------------------------===##

override compiler:=clang++

ifneq ($(target),android)
    PSTL_ARCH += $(KEY)march=native
endif

XHOST_FLAG = -fno-vectorize
CPLUS_FLAGS += $(FQKEY)std=$(stdver)
# XHOST_FLAG = $(KEY)mavx2 -fno-vectorize
# XHOST_FLAG = $(KEY)mavx512f -fno-vectorize
# DYN_LDFLAGS += $(LINK_KEY)c++
# CPLUS_FLAGS += -stdlib=libc++
# CPLUS_FLAGS += -fopenmp-simd //it will be supported in he future version
