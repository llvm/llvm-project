// RUN: not llvm-mc -triple x86_64-unknown-unknown %s 2>&1 | FileCheck %s

.intel_syntax

// CHECK: error: invalid base+index expression
    lea     rdi, [(label + rsi) + rip]
// CHECK: leaq    1(%rax,%rdi), %rdi
    lea     rdi, [(rax + rdi) + 1]
label:
    .quad 42
