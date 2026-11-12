// RUN: not llvm-mc -triple x86_64_lfi %s 2>&1 | FileCheck %s

movq %fs:(%rsp), %rax
// CHECK: error: unsupported addressing mode for %fs access

movq %fs:(%rsp,%rcx,1), %rax
// CHECK: error: unsupported addressing mode for %fs access

movq %fs:foo(%rip), %rax
// CHECK: error: unsupported addressing mode for %fs access

movl %fs:(%edi), %eax
// CHECK: error: unsupported addressing mode for %fs access

cmpq %fs:(%rdi), %r11
// CHECK: error: %fs access reads reserved register %r11

movq %r11, %fs:(%rdi)
// CHECK: error: %fs access reads reserved register %r11
