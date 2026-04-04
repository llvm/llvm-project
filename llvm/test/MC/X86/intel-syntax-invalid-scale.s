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
// CHECK: error: scale factor in address cannot be negative
    lea rax, [rax - 8 * rdx]
// CHECK: error: scale factor in address cannot be negative
    lea rax, [rax - 2 * rdx]
// CHECK: error: scale factor in address cannot be negative
    lea rax, [rdx * 4 + rax - 2 * rcx]
// CHECK: error: scale factor in address cannot be negative
    lea rax, [rax - rdx * 8]
// CHECK: error: scale factor in address cannot be negative
    lea rax, [rax - rdx]
