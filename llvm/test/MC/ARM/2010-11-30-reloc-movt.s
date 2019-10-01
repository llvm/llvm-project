// RUN: llvm-mc  %s -triple=armv7-linux-gnueabi -filetype=obj -o - | \
// RUN:    llvm-readobj -S --sr --sd | FileCheck  %s

	.syntax unified
	.eabi_attribute	6, 10
	.eabi_attribute	8, 1
	.eabi_attribute	9, 2
	.fpu	neon
	.eabi_attribute	20, 1
	.eabi_attribute	21, 1
	.eabi_attribute	23, 3
	.eabi_attribute	24, 1
	.eabi_attribute	25, 1
	.file	"/home/espindola/llvm/llvm/test/CodeGen/ARM/2010-11-30-reloc-movt.ll"
	.text
	.globl	barf
	.align	2
	.type	barf,%function
barf:                                   @ @barf
@ %bb.0:                                @ %entry
	push	{r11, lr}
	movw	r0, :lower16:a
	movt	r0, :upper16:a
	bl	foo
	pop	{r11, pc}
.Ltmp0:
	.size	barf, .Ltmp0-barf



// CHECK:        Section {
// CHECK:          Name: .text
// CHECK:          SectionData (
// CHECK-NEXT:       0000: 00482DE9 000000E3 000040E3 FEFFFFEB
// CHECK-NEXT:       0010: 0088BDE8
// CHECK-NEXT:     )
// CHECK:          Name: .rel.text
// CHECK:          Relocations [
// CHECK-NEXT:       0x4 R_ARM_MOVW_ABS_NC a
// CHECK-NEXT:       0x8 R_ARM_MOVT_ABS
// CHECK-NEXT:       0xC R_ARM_CALL foo
// CHECK-NEXT:     ]
