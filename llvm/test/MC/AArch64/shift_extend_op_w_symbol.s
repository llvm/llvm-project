// RUN: not llvm-mc -triple aarch64-none-linux-gnu < %s > %t1 2> %t2
// RUN: FileCheck < %t1 %s
// RUN: FileCheck --match-full-lines --strict-whitespace --check-prefix=CHECK-ERROR < %t2 %s

        .globl _func
_func:
// CHECK-LABEL: _func

        .set IMM2, 2
        .equ IMM4, 4

// Make sure we can use a symbol with the optionally shift left operand.

        add w1, w2, w3, uxtb #IMM2
        add w4, w5, w6, uxth #IMM4
        add x7, x8, x9, lsl #IMM2
        add w7, w8, w9, uxtw #IMM4
        add x1, x2, x3, uxtx #IMM4

// CHECK: add w1, w2, w3, uxtb #2
// CHECK: add w4, w5, w6, uxth #4
// CHECK: add x7, x8, x9, lsl #2
// CHECK: add w7, w8, w9, uxtw #4
// CHECK: add x1, x2, x3, uxtx #4

        add w1, w2, w3, sxtb #IMM2
        add w4, w5, w6, sxth #IMM4
        add x7, x8, x9, lsl #IMM2
        add w7, w8, w9, sxtw #IMM2
        add x1, x2, x3, sxtx #IMM4

// CHECK: add w1, w2, w3, sxtb #2
// CHECK: add w4, w5, w6, sxth #4
// CHECK: add x7, x8, x9, lsl #2
// CHECK: add w7, w8, w9, sxtw #2
// CHECK: add x1, x2, x3, sxtx #4

        add w1, w2, w3, lsl #IMM3

// CHECK-ERROR:{{.*}}error: expected constant '#imm' after shift specifier
// CHECK-ERROR:        add w1, w2, w3, lsl #IMM3
// CHECK-ERROR:                             ^
