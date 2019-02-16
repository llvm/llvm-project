#===-- icl.inc -----------------------------------------------------------===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.TXT for details.
#
#===----------------------------------------------------------------------===##

vecreport ?= 0 #may be set to [0..7], see https://software.intel.com/en-us/node/522949 for details

XHOST_FLAG = $(QKEY)xHOST
XHOST_FLAG = $(QKEY)vec-
# XHOST_FLAG = $(QKEY)xCORE-AVX2
# XHOST_FLAG = $(QKEY)xSSE4.1
# XHOST_FLAG = $(QKEY)xMIC-AVX512
CPLUS_FLAGS += $(QKEY)opt-assume-safe-padding
CPLUS_FLAGS += $(QKEY)openmp-simd
CPLUS_FLAGS += $(FQKEY)MMD
CPLUS_FLAGS += $(FQKEY)std=$(stdver)
CPLUS_FLAGS += $(QKEY)opt-report:$(vecreport) $(QKEY)opt-report-phase:vec $(QKEY)opt-report-phase:loop

DISABLED_WARNINGS = $(QKEY)diag-disable:2586 #use comma-separated values to specify multiple entries
