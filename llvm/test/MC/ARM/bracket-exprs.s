// RUN: llvm-mc -triple arm-unknown-linux %s | FileCheck %s

// CHECK: .byte 1
.if [~0 >> 63] == 1
.byte 1
.else
.byte 2
.endif

// CHECK: .byte 3
.if 4 * [4 + (3 + [2 * 2] + 1)] == 48
.byte 3
.else
.byte 4
.endif
