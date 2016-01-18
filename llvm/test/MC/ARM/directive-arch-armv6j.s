@ Test the .arch directive for armv6j

@ This test case will check the default .ARM.attributes value for the
@ armv6j architecture.

@ RUN: llvm-mc -triple arm-eabi -filetype asm %s \
@ RUN:   | FileCheck %s -check-prefix CHECK-ASM
@ RUN: llvm-mc -triple arm-eabi -filetype obj %s \
@ RUN:   | llvm-readobj -arm-attributes | FileCheck %s -check-prefix CHECK-ATTR

	.syntax	unified
	.arch	armv6j

@ CHECK-ASM: 	.arch	armv6j

@ CHECK-ATTR: FileAttributes {
@ CHECK-ATTR:   Attribute {
@ CHECK-ATTR:     TagName: CPU_name
@ CHECK-ATTR:     Value: 6J
@ CHECK-ATTR:   }
@ CHECK-ATTR:   Attribute {
@ CHECK-ATTR:     TagName: CPU_arch
@ CHECK-ATTR:     Description: ARM v6
@ CHECK-ATTR:   }
@ CHECK-ATTR:   Attribute {
@ CHECK-ATTR:     TagName: ARM_ISA_use
@ CHECK-ATTR:     Description: Permitted
@ CHECK-ATTR:   }
@ CHECK-ATTR:   Attribute {
@ CHECK-ATTR:     TagName: THUMB_ISA_use
@ CHECK-ATTR:     Description: Thumb-1
@ CHECK-ATTR:   }
@ CHECK-ATTR: }

