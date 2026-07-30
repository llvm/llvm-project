// RUN: not llvm-mc -triple x86_64-unknown-unknown %s 2> %t.err
// RUN: FileCheck < %t.err %s

.intel_syntax

// CHECK: error: base register is 64-bit, but index register is not
    lea rax, [rdi + edx]
