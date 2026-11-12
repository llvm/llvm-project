@ RUN: llvm-mc -triple armv7-eabi -filetype obj -o - %s | llvm-readobj -r - \
@ RUN:     | FileCheck %s
@ RUN: llvm-mc -triple thumbv7-eabi -filetype obj -o - %s | llvm-readobj -r - \
@ RUN:     | FileCheck %s

	.syntax unified

	.section .text.r_arm_abs8

	.byte abs8_0 -128
	.byte abs8_1 +255

@ CHECK: Section {{.*}} .rel.text.r_arm_abs8 {
@ CHECK:   0x0 R_ARM_ABS8 abs8_0
@ CHECK:   0x1 R_ARM_ABS8 abs8_1
@ CHECK: }

	.section .text.r_arm_abs16

	.short abs16_0 -32768
	.short abs16_1 +65535

@ CHECK: Section {{.*}} .rel.text.r_arm_abs16 {
@ CHECK:   0x0 R_ARM_ABS16 abs16_0
@ CHECK:   0x2 R_ARM_ABS16 abs16_1
@ CHECK: }

	.section .text.r_arm_sbrel32

	.word target(sbrel)
	.word target(SBREL)

@ CHECK: Section {{.*}} .rel.text.r_arm_sbrel32 {
@ CHECK:   0x0 R_ARM_SBREL32 target
@ CHECK:   0x4 R_ARM_SBREL32 target
@ CHECK: }

