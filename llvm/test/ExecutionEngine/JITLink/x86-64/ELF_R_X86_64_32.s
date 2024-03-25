# RUN: llvm-mc -triple=x86_64-unknown-linux -position-independent \
# RUN:     -filetype=obj -o %t.o %s
# RUN: llvm-jitlink -noexec -abs X=0x12345678 -check=%s %t.o
# RUN: not llvm-jitlink -noexec -abs X=0x123456789 %t.o 2>&1 | \
# RUN:   FileCheck -check-prefix=CHECK-ERROR %s
#
# Check success and failure cases of R_X86_64_32 handling.

# jitlink-check: *{8}P = X

# CHECK-ERROR: relocation target "X" {{.*}} is out of range of Pointer32 fixup

	.text
	.section	.text.main,"ax",@progbits
	.globl	main
	.p2align	4, 0x90
	.type	main,@function
main:
	xorl	%eax, %eax
	retq
.Lfunc_end0:
	.size	main, .Lfunc_end0-main

	.type	P,@object
	.data
	.globl	P
	.p2align	2, 0x0
P:
	.long	X    # Using long here generates R_X86_64_32.
	.long   0
	.size	P, 8
