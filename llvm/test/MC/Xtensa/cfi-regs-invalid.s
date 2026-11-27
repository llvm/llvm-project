# RUN: not llvm-mc -triple xtensa  < %s 2>&1 | FileCheck %s

.cfi_startproc
.cfi_offset (a0), -4 # CHECK: :[[@LINE]]:14: error: invalid register name
.cfi_offset %foo, 0 # CHECK: :[[@LINE]]:14: error: invalid register name
.cfi_offset ab, -4 # CHECK: :[[@LINE]]:15: error: invalid register name
.cfi_offset f0, -4 # CHECK: :[[@LINE]]:15: error: invalid register name
.cfi_offset sar, -4 # CHECK: :[[@LINE]]:16: error: invalid register name
.cfi_endproc
