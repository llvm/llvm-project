# RUN: not llvm-mc %s -triple=xtensa -filetype=asm 2>&1 | FileCheck %s

.text
.literal_position
.literal .LCPI0_0  a
# CHECK: [[@LINE-1]]:20: error: expected comma
.literal 123, a
# CHECK: [[@LINE-1]]:10: error: literal label must be a symbol
.literal .LCPI1_0,
# CHECK: [[@LINE-1]]:19: error: expected value
