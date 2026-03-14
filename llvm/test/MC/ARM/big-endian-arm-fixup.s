// RUN: llvm-mc -triple=armeb-eabi -mattr v7,vfp2 -filetype=obj < %s | llvm-objdump -s - | FileCheck %s

	.syntax unified
	.text
	.align	2
	.code 32

@ARM::fixup_arm_condbl
.section s_condbl,"ax",%progbits
// CHECK-LABEL: Contents of section s_condbl
// CHECK: 0000 0b000002
 	bleq condbl_label+16
condbl_label:

@ARM::fixup_arm_uncondbl
.section s_uncondbl,"ax",%progbits
// CHECK-LABEL: Contents of section s_uncondbl
// CHECK: 0000 eb000002
 	bl uncond_label+16
uncond_label:

@ARM::fixup_arm_blx
.section s_blx,"ax",%progbits
// CHECK-LABEL: Contents of section s_blx
// CHECK: 0000 fa000002
 	blx blx_label+16
blx_label:

@ARM::fixup_arm_uncondbranch
.section s_uncondbranch,"ax",%progbits
// CHECK-LABEL: Contents of section s_uncondbranch
// CHECK: 0000 ea000003
 	b uncondbranch_label+16
uncondbranch_label:

@ARM::fixup_arm_condbranch
.section s_condbranch,"ax",%progbits
// CHECK-LABEL: Contents of section s_condbranch
// CHECK: 0000 0a000003
 	beq condbranch_label+16
condbranch_label:

@ARM::fixup_arm_pcrel_10
.section s_arm_pcrel_10,"ax",%progbits
// CHECK-LABEL: Contents of section s_arm_pcrel_10
// CHECK: 0000 ed9f0b03
 	vldr d0, arm_pcrel_10_label+16
arm_pcrel_10_label:

@ARM::fixup_arm_ldst_pcrel_12
.section s_arm_ldst_pcrel_12,"ax",%progbits
// CHECK-LABEL: Contents of section s_arm_ldst_pcrel_12
// CHECK: 0000 e59f000c
 	ldr r0, arm_ldst_pcrel_12_label+16
arm_ldst_pcrel_12_label:

@ARM::fixup_arm_adr_pcrel_12
.section s_arm_adr_pcrel_12,"ax",%progbits
// CHECK-LABEL: Contents of section s_arm_adr_pcrel_12
// CHECK: 0000 e28f0010
	adr	r0, arm_adr_pcrel_12_label+20
arm_adr_pcrel_12_label:

@ARM::fixup_arm_adr_pcrel_10_unscaled
.section s_arm_adr_pcrel_10_unscaled,"ax",%progbits
// CHECK-LABEL: Contents of section s_arm_adr_pcrel_10_unscaled
// CHECK: 0000 e1cf01d4
	ldrd	r0, r1, arm_adr_pcrel_10_unscaled_label+24
arm_adr_pcrel_10_unscaled_label:

@ARM::fixup_arm_movw_lo16
.section s_movw,"ax",%progbits
// CHECK-LABEL: Contents of section s_movw
// CHECK: 0000 e3000008
	movw	r0, :lower16:(some_label+8)

@ARM::fixup_arm_movt_hi16
.section s_movt,"ax",%progbits
// CHECK-LABEL: Contents of section s_movt
// CHECK: 0000 e34f0ffc
	movt	r0, :upper16:GOT-(movt_label)
movt_label:

@FK_Data_1
.section s_fk_data_1
// CHECK-LABEL: Contents of section s_fk_data_1
// CHECK: 0000 01
fk_data1_l_label:
.byte fk_data1_h_label-fk_data1_l_label
fk_data1_h_label:

@FK_Data_2
.section s_fk_data_2
// CHECK-LABEL: Contents of section s_fk_data_2
// CHECK: 0000 0002
fk_data2_l_label:
.short fk_data2_h_label-fk_data2_l_label
fk_data2_h_label:

@FK_Data_4
.section s_fk_data_4
// CHECK-LABEL: Contents of section s_fk_data_4
// CHECK: 0000 00000004
fk_data4_l_label:
.long fk_data4_h_label-fk_data4_l_label
fk_data4_h_label:

