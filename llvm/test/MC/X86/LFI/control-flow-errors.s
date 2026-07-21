// RUN: not llvm-mc -triple x86_64_lfi %s 2>&1 | FileCheck %s

// Masking the target in place would clobber a register the sandbox relies on.

jmpq *%r14
// CHECK: error: indirect branch through reserved register

callq *%r15
// CHECK: error: indirect branch through reserved register

jmpq *%rsp
// CHECK: error: indirect branch through reserved register

// Far branches cannot be sandboxed.

ljmpq *(%rax)
// CHECK: error: unsupported indirect branch

lcallq *(%rax)
// CHECK: error: unsupported indirect branch

// Only 64-bit near returns are supported.

retw
// CHECK: error: unsupported return instruction

retw $8
// CHECK: error: unsupported return instruction

lret
// CHECK: error: unsupported return instruction

iretq
// CHECK: error: unsupported return instruction
