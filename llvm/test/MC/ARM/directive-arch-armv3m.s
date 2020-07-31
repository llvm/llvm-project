@ Test the .arch directive for armv3m

@ This test case will check the default .ARM.attributes value for the
@ armv3m architecture.

@ RUN: llvm-mc -triple arm-eabi -filetype asm %s \
@ RUN:   | FileCheck %s -check-prefix CHECK-ASM
@ RUN: llvm-mc -triple arm-eabi -filetype obj %s \
@ RUN:   | llvm-readobj --arch-specific - | FileCheck %s -check-prefix CHECK-ATTR

	.syntax	unified
	.arch	armv3m

@ CHECK-ASM: 	.arch	armv3m

@ CHECK-ATTR: FileAttributes {
@ CHECK-ATTR:   Attribute {
@ CHECK-ATTR:     TagName: CPU_name
@ CHECK-ATTR:     Value: 3M
@ CHECK-ATTR:   }
@ CHECK-ATTR:   Attribute {
@ CHECK-ATTR:     TagName: CPU_arch
@ CHECK-ATTR:     Description: Pre-v4
@ CHECK-ATTR:   }
@ CHECK-ATTR:   Attribute {
@ CHECK-ATTR:     TagName: ARM_ISA_use
@ CHECK-ATTR:     Description: Permitted
@ CHECK-ATTR:   }
@ CHECK-ATTR: }

