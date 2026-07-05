// RUN: llvm-mc < %s -triple=aarch64-none-linux-gnu -filetype=obj | llvm-readobj -r - | FileCheck %s

	.file	"<stdin>"
	.text
	.globl	test_jumptable
	.type	test_jumptable,@function
test_jumptable:                         // @test_jumptable
	.cfi_startproc
// %bb.0:
	ubfx	w1, w0, #0, #32
	cmp w0, #4
	b.hi .LBB0_3
// %bb.1:
	adrp	x0, .LJTI0_0
	add	x0, x0, #:lo12:.LJTI0_0
	ldr	x0, [x0, x1, lsl #3]
	br	x0
.LBB0_2:                                // %lbl1
	movz	x0, #1
	ret
.LBB0_3:                                // %def
	mov	 x0, xzr
	ret
.LBB0_4:                                // %lbl2
	movz	x0, #2
	ret
.LBB0_5:                                // %lbl3
	movz	x0, #4
	ret
.LBB0_6:                                // %lbl4
	movz	x0, #8
	ret
.Ltmp0:
	.size	test_jumptable, .Ltmp0-test_jumptable
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.align	3
.LJTI0_0:
	.xword	.LBB0_2
	.xword	.LBB0_4
	.xword	.LBB0_5
	.xword	.LBB0_3
	.xword	.LBB0_6



// First make sure we get a page/lo12 pair in .text to pick up the jump-table

// CHECK:      Relocations [
// CHECK:        Section ({{[0-9]+}}) .rela.text {
// CHECK-NEXT:     0x{{[0-9,A-F]+}} R_AARCH64_ADR_PREL_PG_HI21 .rodata
// CHECK-NEXT:     0x{{[0-9,A-F]+}} R_AARCH64_ADD_ABS_LO12_NC .rodata
// CHECK:        }

// Also check the targets in .rodata are relocated
// CHECK:        Section ({{[0-9]+}}) .rela.rodata {
// CHECK-NEXT:     0x{{[0-9,A-F]+}} R_AARCH64_ABS64 .text
// CHECK:        }
// CHECK:      ]
