// RUN: not llvm-mc -triple x86_64_lfi -mattr=+clzero,+movdir64b,+enqcmd %s 2>&1 | FileCheck %s

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

movl %eax, %esp
// CHECK: error: unsupported modification of the stack pointer

addl $8, %esp
// CHECK: error: unsupported modification of the stack pointer

xchgl %eax, %esp
// CHECK: error: unsupported modification of the stack pointer

shll $1, %esp
// CHECK: error: unsupported modification of the stack pointer

cmpxchgl %ecx, %esp
// CHECK: error: unsupported modification of the stack pointer

xchgq %rsp, %fs:0
// CHECK: error: unsupported modification of the stack pointer

movl %fs:0, %esp
// CHECK: error: unsupported modification of the stack pointer

btq %rax, (%rdi)
// CHECK: error: unsupported bit test with a register bit offset

btsq %rax, (%rdi)
// CHECK: error: unsupported bit test with a register bit offset

lock btsq %rax, (%rdi)
// CHECK: error: unsupported bit test with a register bit offset

btrl %eax, %fs:(%rdi)
// CHECK: error: unsupported bit test with a register bit offset

btq %rax, %fs:0
// CHECK: error: unsupported bit test with a register bit offset

btsq %rax, %fs:0
// CHECK: error: unsupported bit test with a register bit offset

gs movq (%rsp), %rax
// CHECK: error: unsupported instruction prefix

fs movq (%rsp), %rax
// CHECK: error: unsupported instruction prefix

addr32 movq (%rsp), %rax
// CHECK: error: unsupported instruction prefix

rex64 movq %rdi, %rsp
// CHECK: error: unsupported instruction prefix

data16 addq $8, %rsp
// CHECK: error: unsupported instruction prefix

data16
leave
// CHECK: error: unsupported instruction prefix

data16
retq $8
// CHECK: error: unsupported instruction prefix

lock
jmpq *(%rdi)
// CHECK: error: unsupported instruction prefix

lock jmpq *(%rdi)
// CHECK: error: unsupported lock prefix

movl (%edi), %eax
// CHECK: error: unsupported 32-bit address size

movq 8(%eax,%ecx,4), %rbx
// CHECK: error: unsupported 32-bit address size

movq (%eax,%eiz), %rbx
// CHECK: error: unsupported 32-bit address size

movq (,%eiz,1), %rbx
// CHECK: error: unsupported 32-bit address size

movq (%esp), %rax
// CHECK: error: unsupported 32-bit address size

movq foo(%eip), %rax
// CHECK: error: unsupported 32-bit address size

jmpq *(%eax)
// CHECK: error: unsupported 32-bit address size

vgatherdpd %xmm3, (%eax,%xmm2,8), %xmm0
// CHECK: error: unsupported 32-bit address size

movsq (%esi), %es:(%edi)
// CHECK: error: unsupported address size for string instruction

stosq %rax, %es:(%edi)
// CHECK: error: unsupported address size for string instruction

movsq %fs:(%rsi), %es:(%rdi)
// CHECK: error: unsupported segment override for string instruction

lodsq %fs:(%rsi), %rax
// CHECK: error: unsupported segment override for string instruction

enter $16, $2
// CHECK: error: unsupported nesting level for enter

clzero
// CHECK: error: unsupported memory access

movdir64b (%rax), %rbx
// CHECK: error: unsupported memory access

enqcmd (%rax), %rbx
// CHECK: error: unsupported memory access

movdir64b %fs:0, %rbx
// CHECK: error: unsupported memory access

enqcmd %fs:0, %rbx
// CHECK: error: unsupported memory access

.code32

lock
// CHECK: error: LFI only supports 64-bit mode

.code64

lock
// CHECK: error: unsupported instruction prefix
