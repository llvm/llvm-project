@ Check multiple .fpu directives.

@ The later .fpu directive should overwrite the earlier one.
@ We also check here that all the .fpu directives that we expect to work do work

@ RUN: llvm-mc -triple arm-eabi -filetype obj %s | llvm-readobj --arm-attributes \
@ RUN:   | FileCheck %s -check-prefix CHECK-ATTR

	.fpu none
	.fpu vfp
	.fpu vfpv2
	.fpu vfpv3
	.fpu vfpv3-fp16
	.fpu vfpv3-d16
	.fpu vfpv3-d16-fp16
	.fpu vfpv3xd
	.fpu vfpv3xd-fp16
	.fpu vfpv4
	.fpu vfpv4-d16
	.fpu fpv4-sp-d16
	.fpu fpv5-d16
	.fpu fpv5-sp-d16
	.fpu fp-armv8
	.fpu neon
	.fpu neon-fp16
	.fpu neon-vfpv4
	.fpu neon-fp-armv8
	.fpu crypto-neon-fp-armv8
	.fpu softvfp

	.fpu vfpv4

@ CHECK-ATTR: FileAttributes {
@ CHECK-ATTR:   Attribute {
@ CHECK-ATTR:     TagName: FP_arch
@ CHECK-ATTR:     Description: VFPv4
@ CHECK-ATTR:   }
@ CHECK-ATTR: }

