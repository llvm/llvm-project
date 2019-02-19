#===-- clang.inc ---------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
