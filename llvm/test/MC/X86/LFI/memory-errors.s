// RUN: not llvm-mc -triple x86_64_lfi %s 2>&1 | FileCheck %s

movq %gs:(%rax), %rbx
// CHECK: error: invalid use of reserved segment register %gs

movq %gs:8, %rbx
// CHECK: error: invalid use of reserved segment register %gs

movw %ax, %gs
// CHECK: error: illegal modification of reserved LFI register

wrgsbase %rax
// CHECK: error: illegal modification of reserved segment base

wrgsbasel %eax
// CHECK: error: illegal modification of reserved segment base

wrfsbase %rax
// CHECK: error: illegal modification of reserved segment base

wrfsbasel %eax
// CHECK: error: illegal modification of reserved segment base

movabsq 0x123456789abcdef, %rax
// CHECK: error: unsupported memory access

movabsb %al, 0x123456789abcdef
// CHECK: error: unsupported memory access

xlatb
// CHECK: error: unsupported memory access

maskmovdqu %xmm1, %xmm0
// CHECK: error: unsupported memory access

xchgq %rax, %rsp
// CHECK: error: unsupported modification of the stack pointer

shlq $1, %rsp
// CHECK: error: unsupported modification of the stack pointer

movw %ax, %sp
// CHECK: error: unsupported modification of the stack pointer

.code32

lock
// CHECK: error: LFI only supports 64-bit mode

.code64

lock
// CHECK: error: unsupported instruction prefix
