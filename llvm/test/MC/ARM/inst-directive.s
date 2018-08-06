@ RUN: llvm-mc %s -triple=armv7-linux-gnueabi -filetype=obj -o - \
@ RUN:   | llvm-readobj -s -sd | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-LE

@ RUN: llvm-mc %s -triple=armebv7-linux-gnueabi -filetype=obj -o - \
@ RUN:   | llvm-readobj -s -sd | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-BE

	.syntax unified

@-------------------------------------------------------------------------------
@ arm_inst
@-------------------------------------------------------------------------------
	.arm

	.section	.inst.arm_inst

	.align	2
	.global	arm_inst
	.type	arm_inst,%function
arm_inst:
	.inst 0xdefe

@ CHECK: Section {
@ CHECK:   Name: .inst.arm_inst
@ CHECK:   SectionData (
@ CHECK-LE-NEXT:     0000: FEDE0000
@ CHECK-BE-NEXT:     0000: 0000DEFE
@ CHECK-NEXT:   )

@-------------------------------------------------------------------------------
@ thumb_inst_n
@-------------------------------------------------------------------------------
	.thumb

	.section	.inst.thumb_inst_n

	.align	2
	.global	thumb_inst_n
	.type	thumb_inst_n,%function
thumb_inst_n:
	.inst.n 0xdefe

@ CHECK: Section {
@ CHECK:   Name: .inst.thumb_inst_n
@ CHECK:   SectionData (
@ CHECK-LE-NEXT:     0000: FEDE
@ CHECK-BE-NEXT:     0000: DEFE
@ CHECK-NEXT:   )

@-------------------------------------------------------------------------------
@ thumb_inst_w
@-------------------------------------------------------------------------------
	.thumb

	.section	.inst.thumb_inst_w

	.align	2
	.global	thumb_inst_w
	.type	thumb_inst_w,%function
thumb_inst_w:
	.inst.w 0x12345678

@ CHECK: Section {
@ CHECK:   Name: .inst.thumb_inst_w
@ CHECK:   SectionData (
@ CHECK-LE-NEXT:     0000: 34127856
@ CHECK-BE-NEXT:     0000: 12345678
@ CHECK-NEXT:   )

@-------------------------------------------------------------------------------
@ thumb_inst_w
@-------------------------------------------------------------------------------
	.thumb

	.section	.inst.thumb_inst_inst

	.align	2
	.global	thumb_inst_inst
	.type	thumb_inst_inst,%function
thumb_inst_inst:
	.inst.w 0xf2400000, 0xf2c00000

@ CHECK: Section {
@ CHECK:   Name: .inst.thumb_inst_inst
@ CHECK:   SectionData (
@ CHECK-LE-NEXT:     0000: 40F20000 C0F20000
@ CHECK-BE-NEXT:     0000: F2400000 F2C00000
@ CHECK-NEXT:   )

