// RUN: llvm-mc -triple aarch64-none-linux-gnu %s -filetype=obj -o %t
// RUN: llvm-objdump --no-print-imm-hex --triple aarch64-none-linux-gnu -Dr %t | FileCheck %s

0:
.skip 4
1:
mov x0, 1b - 0b
// CHECK: mov x0, #4
mov x0, 0b - 1b
// CHECK: mov x0, #-4
mov x0, 0b - 0b
// CHECK: mov x0, #0
mov x0, 1b - 2 - 0b + 6
// CHECK: mov x0, #8
mov x0, #:abs_g0_s:1b
// CHECK: mov x0, #0
// CHECK-NEXT: R_AARCH64_MOVW_SABS_G0	.text+0x4
