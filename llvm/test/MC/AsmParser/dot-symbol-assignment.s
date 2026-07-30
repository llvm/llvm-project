# RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s

	.extern	start

# CHECK: .org 1024, 0
. = 0x400
	lgdt	0x400 + 0x100

	ljmpl	$0x08, $(0x400 + 0x150)


# CHECK: .org 1280, 0
. = 0x400 + 0x100
	.word	(3*8)-1
	.quad	(0x400 + 0x110)

# CHECK: .org 1296, 0
. = 0x400 + 0x110
	.quad	0x0
	.quad	0x0020980000000000
	.quad	0x0000900000000000

	.code64

# CHECK: .org 1360, 0
. = 0x400 + 0x150
	movabsq	$start, %rcx
	jmp	*%rcx


. = 0x300
