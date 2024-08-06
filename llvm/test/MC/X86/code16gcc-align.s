# RUN: llvm-mc -filetype=obj -triple=i386-unknown-unknown-code16 %s | llvm-objdump --triple=i386-unknown-unknown-code16 -d - | FileCheck %s

# Ensure that the "movzbl" is aligned such that the prefixes 0x67 0x66 are
# properly included in the "movz" instruction.

# CHECK-LABEL: <test>:
# CHECK:           1c: 8d b4 00 00                  	leaw	(%si), %si
# CHECK-NEXT:      20: 66 90                        	nop
# CHECK-NEXT:      22: 66 89 c7                     	movl	%eax, %edi
# CHECK-NEXT:      25: 66 31 db                     	xorl	%ebx, %ebx
# CHECK-NEXT:      28: 8d b4 00 00                  	leaw	(%si), %si
# CHECK-NEXT:      2c: 8d b4 00 00                      leaw	(%si), %si
# CHECK-NEXT:      30: 67 66 0f b6 0c 1e            	movzbl	(%esi,%ebx), %ecx
# CHECK-NEXT:      36: 66 e8 14 00 00 00            	calll	0x50 <called>
# CHECK-NEXT:      3c: 8d 74 00                     	leaw	(%si), %si

# CHECK-LABEL: <called>:
# CHECK-NEXT:      50: 90                           	nop
# CHECK-NEXT:      51: 66 c3                        	retl

	.text
	.code16gcc
	.globl	test
	.p2align	4
	.type	test,@function
test:
	.nops	34
	movl	%eax, %edi
	xorl	%ebx, %ebx
	.p2align	4
	movzbl	(%esi,%ebx), %ecx
	calll	called
	.nops	3
	retl

	.p2align	4
	.type	called,@function
called:
	.nops	1
	retl
