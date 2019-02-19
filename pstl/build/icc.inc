#===-- icc.inc -----------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

vecreport ?= 0 #may be set to [0..7], see https://software.intel.com/en-us/node/522949 for details

XHOST_FLAG = $(KEY)xHOST -no-vec
CPLUS_FLAGS += $(QKEY)opt-assume-safe-padding

# XHOST_FLAG = $(KEY)xCORE-AVX2 -no-vec
# XHOST_FLAG = $(KEY)xSSE4.1 -no-vec
# XHOST_FLAG = $(KEY)xMIC-AVX512 -no-vec

CPLUS_FLAGS += $(QKEY)openmp-simd
CPLUS_FLAGS += $(FQKEY)MMD
CPLUS_FLAGS += $(FQKEY)std=$(stdver)
CPLUS_FLAGS +=  $(QKEY)opt-report=$(vecreport) $(QKEY)opt-report-phase vec

OPTIMIZATION_DISABLED_FLAGS += $(KEY)debug inline-debug-info
