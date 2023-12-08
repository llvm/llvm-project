@ RUN: llvm-mc -triple armv7-eabi %s | FileCheck %s

	.data

unaligned:
	.byte 1
	.align

@ CHECK-LABEL: unaligned
@ CHECK-NEXT:	.byte 1
@ CHECK-NEXT:	.p2align 2

aligned:
	.long 0x1d10c1e5
	.align

@ CHECK-LABEL: aligned
@ CHECK-NEXT:	.long 487637477
@ CHECK-NEXT:	.p2align 2

trailer:
	.long 0xd1ab011c
	.align 2

@ CHECK-LABEL: trailer
@ CHECK-NEXT:	.long 3517645084
@ CHECK-NEXT:	.p2align 2

