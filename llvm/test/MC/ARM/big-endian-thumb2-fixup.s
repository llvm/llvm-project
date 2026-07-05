// RUN: llvm-mc -triple=thumbeb-eabi -mattr v7,vfp2 -filetype=obj < %s | llvm-objdump -s - | FileCheck %s

	.syntax unified
	.text
	.align	2

@ARM::fixup_t2_movw_lo16
.section s_movw,"ax",%progbits
// CHECK-LABEL: Contents of section s_movw
// CHECK: 0000 f2400008
	movw	r0, :lower16:(some_label+8)

@ARM::fixup_t2_movt_hi16
.section s_movt,"ax",%progbits
// CHECK-LABEL: Contents of section s_movt
// CHECK: 0000 f6cf70fc
	movt	r0, :upper16:GOT-(movt_label)
movt_label:

@ARM::fixup_t2_uncondbranch
.section s_uncondbranch,"ax",%progbits
// CHECK-LABEL: Contents of section s_uncondbranch
// CHECK: 0000 f000b801 bf00
 	b.w uncond_label
	nop
uncond_label:

@ARM::fixup_t2_condbranch
.section s_condbranch,"ax",%progbits
// CHECK-LABEL: Contents of section s_condbranch
// CHECK: 0000 f0008001 bf00
 	beq.w cond_label
	nop
cond_label:

@ARM::fixup_t2_ldst_precel_12
.section s_ldst_precel_12,"ax",%progbits
 	ldr.w r0, ldst_precel_12_label
	nop
	nop
ldst_precel_12_label:

@ARM::fixup_t2_adr_pcrel_12
.section s_adr_pcrel_12,"ax",%progbits
 	adr.w r0, adr_pcrel_12_label
	nop
	nop
adr_pcrel_12_label:

@ARM::fixup_t2_so_imm
.section s_t2_so_imm,"ax",%progbits
// CHECK-LABEL: Contents of section s_t2_so_imm
// CHECK: 0000 f1033337
	add r3, r3,val
.equ val,0x37373737
