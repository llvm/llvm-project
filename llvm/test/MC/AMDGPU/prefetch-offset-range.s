// RUN: not llvm-mc -triple=amdgpu12.00 -filetype=obj -o /dev/null %s 2>&1 | FileCheck %s

// CHECK: prefetch-offset-range.s:[[#@LINE+4]]:24: error: prefetch offset exceeds 24-bit signed range

// The displacement is a signed 24-bit byte offset from the next instruction.
// 8388608 is one past the maximum of 8388607.
s_prefetch_inst_pc_rel far, s10, 7
.space 8388608
far:
