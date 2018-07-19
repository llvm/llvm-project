@ RUN: llvm-mc %s -triple=armv7-linux-gnueabi -filetype=obj -o - \
@ RUN:   | llvm-readobj -s -sd | FileCheck %s

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
@ CHECK-NEXT:     0000: FEDE0000
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
@ CHECK-NEXT:     0000: FEDE
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
	.inst.w 0x00000000

@ CHECK: Section {
@ CHECK:   Name: .inst.thumb_inst_w
@ CHECK:   SectionData (
@ CHECK-NEXT:     0000: 00000000
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
@ CHECK-NEXT:     0000: 40F20000 C0F20000
@ CHECK-NEXT:   )

