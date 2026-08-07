// RUN: not llvm-mc -triple x86_64-unknown-unknown %s 2> %t.err
// RUN: FileCheck < %t.err %s

.intel_syntax

// CHECK: error: scale factor in address must be 1, 2, 4 or 8
    lea rax, [rdi + rdx*64]
// CHECK: error: scale factor in address must be 1, 2, 4 or 8
    lea rax, [rdi + rdx*32]
// CHECK: error: scale factor in address must be 1, 2, 4 or 8
    lea rax, [rdi + rdx*16]
// CHECK: error: Scale can't be negative
    lea rax, [rdi + rdx*-8]
// CHECK: error: scale factor in address must be 1, 2, 4 or 8
    lea rax, [rdi + -1*rdx]
// CHECK: [[#@LINE+3]]:19: error: Scale can't be negative
// CHECK-NEXT:     lea rax, [rax - 8 * rdx]
// CHECK-NEXT:                   ^
    lea rax, [rax - 8 * rdx]
// CHECK: [[#@LINE+3]]:19: error: Scale can't be negative
// CHECK-NEXT:     lea rax, [rax - 2 * rdx]
// CHECK-NEXT:                   ^
    lea rax, [rax - 2 * rdx]
// CHECK: [[#@LINE+3]]:19: error: Scale can't be negative
// CHECK-NEXT:     lea rax, [rax - rdx * 8]
// CHECK-NEXT:                   ^
    lea rax, [rax - rdx * 8]
// CHECK: [[#@LINE+3]]:19: error: Scale can't be negative
// CHECK-NEXT:     lea rax, [rax - rdx]
// CHECK-NEXT:                   ^
    lea rax, [rax - rdx]

// Registers are tracked separately from the expression evaluator, so scaling a
// parenthesised group holding one used to drop the scale silently rather than
// reject the expression.

// CHECK: [[#@LINE+3]]:27: error: cannot multiply a subexpression containing a register
// CHECK-NEXT:     lea rax, [(rdi + rdx) * 4]
// CHECK-NEXT:                           ^
    lea rax, [(rdi + rdx) * 4]
// CHECK: [[#@LINE+3]]:20: error: cannot multiply a subexpression containing a register
// CHECK-NEXT:     lea rax, [4 * (rdi + rdx)]
// CHECK-NEXT:                    ^
    lea rax, [4 * (rdi + rdx)]
// CHECK: [[#@LINE+3]]:21: error: cannot multiply a subexpression containing a register
// CHECK-NEXT:     lea rax, [(rdi) * 4]
// CHECK-NEXT:                     ^
    lea rax, [(rdi) * 4]
// CHECK: [[#@LINE+3]]:21: error: scale must be an immediate
// CHECK-NEXT:     lea rax, [rdi * (1 + 1)]
// CHECK-NEXT:                     ^
    lea rax, [rdi * (1 + 1)]
// CHECK: [[#@LINE+3]]:25: error: cannot divide a subexpression containing a register
// CHECK-NEXT:     lea rax, [(4 * rdi) / 2]
// CHECK-NEXT:                         ^
    lea rax, [(4 * rdi) / 2]
