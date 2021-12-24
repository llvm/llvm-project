// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o %t
// RUN: llvm-jitlink -noexec %t

.section	.foo,"ax"
.globl  zero
zero:


.text
.globl main
main:
nop
