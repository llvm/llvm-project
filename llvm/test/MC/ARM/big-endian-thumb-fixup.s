// RUN: llvm-mc -triple=armeb-eabi -mattr v7,vfp2 -filetype=obj < %s | llvm-objdump -s - | FileCheck %s

	.syntax unified
	.text
	.align	2
	.code 16
	.thumb_func

@ARM::fixup_arm_thumb_bl
.section s_thumb_bl,"ax",%progbits
// CHECK-LABEL: Contents of section s_thumb_bl
// CHECK: 0000 f000f801
 	bl thumb_bl_label
	nop
thumb_bl_label:

@ARM::fixup_arm_thumb_blx
// CHECK-LABEL: Contents of section s_thumb_bl
// CHECK: 0000 f000e802
.section s_thumb_blx,"ax",%progbits
 	blx thumb_blx_label+8
thumb_blx_label:

@ARM::fixup_arm_thumb_br
.section s_thumb_br,"ax",%progbits
// CHECK-LABEL: Contents of section s_thumb_br
// CHECK: 0000 e000bf00
 	b thumb_br_label
	nop
thumb_br_label:

@ARM::fixup_arm_thumb_bcc
.section s_thumb_bcc,"ax",%progbits
// CHECK-LABEL: Contents of section s_thumb_bcc
// CHECK: 0000 d000bf00
 	beq thumb_bcc_label
	nop
thumb_bcc_label:

@ARM::fixup_arm_thumb_cb
.section s_thumb_cb,"ax",%progbits
// CHECK-LABEL: Contents of section s_thumb_cb
// CHECK: 0000 b100bf00
 	cbz r0, thumb_cb_label
	nop
thumb_cb_label:

@ARM::fixup_arm_thumb_cp
.section s_thumb_cp,"ax",%progbits
// CHECK-LABEL: Contents of section s_thumb_cp
// CHECK: 0000 4801bf00
 	ldr r0, =thumb_cp_label
	nop
	nop
thumb_cp_label:

@ARM::fixup_arm_thumb_adr_pcrel_10
.section s_thumb_adr_pcrel_10,"ax",%progbits
// CHECK-LABEL: Contents of section s_thumb_adr_pcrel_10
// CHECK: 0000 a000bf00
	adr r0, thumb_adr_pcrel_10_label
	nop
thumb_adr_pcrel_10_label:

