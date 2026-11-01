# RUN: llvm-mc -triple=x86_64-unknown-linux -position-independent \
# RUN:     -filetype=obj -o %t.o %s
# RUN: llvm-jitlink -noexec -abs X=0x12 -check=%s %t.o
# RUN: not llvm-jitlink -noexec -abs X=0x123 %t.o 2>&1 | \
# RUN:   FileCheck -check-prefix=CHECK-ERROR %s
#
# Check success and failure cases of R_X86_64_16 handling.

# jitlink-check: *{8}P = X

# The fixup address is the address being patched, which is P's block address
# because the fixup is at offset 0. It must not be the relocation target.
# CHECK-ERROR: relocation target 0x123 (X) is out of range of Pointer8 fixup at address [[FIXUP:0x[0-9a-f]+]] (P, [[FIXUP]] +

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
P:
	.byte	X    # Using byte here generates R_X86_64_8.
	.byte   0
	.byte   0
	.byte   0
	.byte   0
	.byte   0
	.byte   0
	.byte   0
	.size	P, 8
