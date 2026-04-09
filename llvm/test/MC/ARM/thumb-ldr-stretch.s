@ RUN: llvm-mc -triple thumbv7m -filetype=obj -o %t %s
@ RUN: llvm-objdump -d --no-show-raw-insn --triple=thumbv7m %t | FileCheck %s

@ The three "b external" instructions relax from 2-byte tB to 4-byte t2B
@ (unresolved fixup). The cumulative size growth must not cause the tLDRpci
@ instructions to appear misaligned and spuriously relax to 4-byte t2LDRpci.
@ The .p2align 2 before the constant pool absorbs the upstream growth, keeping
@ the targets 4-byte aligned.
@
@ If tLDRpci spuriously widens, the extra bytes push the cbz target past the
@ 126-byte range, causing "out of range pc-relative fixup value".

@ CHECK-LABEL: <fn>:
@ CHECK:        0: cbz r0, 0x7e
@ CHECK:       4c: ldr r2, [pc, #0x30]
@ CHECK-NOT:       ldr.w
@ CHECK:       5e: ldr r2, [pc, #0x24]
@ CHECK-NOT:       ldr.w
@ CHECK:       70: ldr r2, [pc, #0x14]
@ CHECK-NOT:       ldr.w
@ CHECK:       7e: nop

	.syntax	unified
	.text
	.globl	fn
	.type	fn,%function
	.thumb_func
fn:
	cbz	r0, .LBB0_11
	.rept 6
	nop
	.endr
	b	.LBB0_11
	.rept 6
	nop
	.endr
	b	external
	.rept 14
	nop
	.endr
	b	.LBB0_11
	.rept 7
	nop
	.endr
	ldr.n	r2, .LCPI0_3
	.rept 6
	nop
	.endr
	b	external
	ldr	r2, .LCPI0_4
	.rept 6
	nop
	.endr
	b	external
	ldr.n	r2, .LCPI0_5
	.rept 6
	nop
	.endr
.LBB0_11:
	.p2align	2
.LCPI0_3:
	.long	4
.LCPI0_4:
	.long	5
.LCPI0_5:
	.long	6
