// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -r - | FileCheck %s

// Test that this produces a R_X86_64_PLT32 pointing to .Lfoo
// instead of the section symbol.

.section .text.a,"ax",@progbits
jmp .Lfoo

.text
ret
.Lfoo:
ret

// CHECK: R_X86_64_PLT32 .Lfoo 0xFFFFFFFFFFFFFFFC
