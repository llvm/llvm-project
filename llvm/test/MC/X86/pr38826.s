// RUN: llvm-mc %s -triple i386-unknown-unknown

// Make sure %eip is allowed as a register in cfi directives in 32-bit mode

 .text
 .align 4
 .globl foo

foo:
 .cfi_startproc

 movl (%edx), %ecx
 movl 4(%edx), %ebx
 movl 8(%edx), %esi
 movl 12(%edx), %edi
 movl 16(%edx), %ebp
 .cfi_def_cfa %edx, 0
 .cfi_offset %eip, 24
 .cfi_register %esp, %ecx
 movl %ecx, %esp

 jmp *24(%edx)

 .cfi_endproc
