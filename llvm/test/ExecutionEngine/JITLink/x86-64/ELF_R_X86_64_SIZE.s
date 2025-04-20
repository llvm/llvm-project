# Checks that JITLink is able to handle R_X86_64_SIZE32/R_X86_64_SIZE64 relocations.
# RUN: llvm-mc -triple=x86_64-unknown-linux -position-independent \
# RUN:     -filetype=obj -o %t.1.o %s
# RUN: llvm-jitlink -noexec %t.1.o

# Checks that JITLink emits an error message when the fixup cannot fit into a 32-bit value.
# RUN: llvm-mc -triple=x86_64-unknown-linux -position-independent --defsym=OVERFLOW=1 \
# RUN:     -filetype=obj -o %t.2.o %s
# RUN: not llvm-jitlink -noexec %t.2.o 2>&1 | FileCheck %s
# CHECK: llvm-jitlink error: In graph {{.*}}, section .text: relocation target "main" at address {{.*}} is out of range of Size32 fixup at {{.*}} (main, {{.*}})

	.text
	.globl	main
	.type	main,@function
main:
	xorl	%eax, %eax
	movq	main@SIZE + 2, %rbx  # Generate R_X86_64_SIZE32 relocation.
.ifndef OVERFLOW
	movl	main@SIZE + 1, %ebx  # Generate R_X86_64_SIZE32 relocation.
.else
	movl	main@SIZE - 32, %ebx # Generate R_X86_64_SIZE32 relocation whose fixup overflows.
.endif	
	retq
	.size	main, .-main

	.data
	.quad	main@SIZE + 1 # Generate R_X86_64_SIZE64 relocation.
