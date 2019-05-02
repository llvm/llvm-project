@ Test the .arch directive for armv6-m

@ This test case will check the default .ARM.attributes value for the
@ armv6-m architecture.

@ RUN: llvm-mc -triple arm-eabi -filetype asm %s \
@ RUN:   | FileCheck %s -check-prefix CHECK-ASM
@ RUN: llvm-mc -triple arm-eabi -filetype obj %s \
@ RUN:   | llvm-readobj --arm-attributes | FileCheck %s -check-prefix CHECK-ATTR

	.syntax	unified
	.arch	armv6-m

@ CHECK-ASM: 	.arch	armv6-m

@ CHECK-ATTR: FileAttributes {
@ CHECK-ATTR:   Attribute {
@ CHECK-ATTR:     TagName: CPU_name
@ CHECK-ATTR:     Value: 6-M
@ CHECK-ATTR:   }
@ CHECK-ATTR:   Attribute {
@ CHECK-ATTR:     TagName: CPU_arch
@ CHECK-ATTR:     Description: ARM v6-M
@ CHECK-ATTR:   }
@ CHECK-ATTR:   Attribute {
@ CHECK-ATTR:     TagName: THUMB_ISA_use
@ CHECK-ATTR:     Description: Thumb-1
@ CHECK-ATTR:   }
@ CHECK-ATTR: }

