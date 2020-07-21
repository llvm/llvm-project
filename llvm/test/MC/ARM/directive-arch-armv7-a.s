@ Test the .arch directive for armv7-a

@ This test case will check the default .ARM.attributes value for the
@ armv7-a architecture.

@ RUN: llvm-mc -triple arm-eabi -filetype asm %s \
@ RUN:   | FileCheck %s -check-prefix CHECK-ASM
@ RUN: llvm-mc -triple arm-eabi -filetype obj %s \
@ RUN:   | llvm-readobj --arch-specific - | FileCheck %s -check-prefix CHECK-ATTR

	.syntax	unified
	.arch	armv7-a

@ CHECK-ASM: 	.arch	armv7-a

@ CHECK-ATTR: FileAttributes {
@ CHECK-ATTR:   Attribute {
@ CHECK-ATTR:     TagName: CPU_name
@ CHECK-ATTR:     Value: 7-A
@ CHECK-ATTR:   }
@ CHECK-ATTR:   Attribute {
@ CHECK-ATTR:     TagName: CPU_arch
@ CHECK-ATTR:     Description: ARM v7
@ CHECK-ATTR:   }
@ CHECK-ATTR:   Attribute {
@ CHECK-ATTR:     TagName: CPU_arch_profile
@ CHECK-ATTR:     Description: Application
@ CHECK-ATTR:   }
@ CHECK-ATTR:   Attribute {
@ CHECK-ATTR:     TagName: ARM_ISA_use
@ CHECK-ATTR:     Description: Permitted
@ CHECK-ATTR:   }
@ CHECK-ATTR:   Attribute {
@ CHECK-ATTR:     TagName: THUMB_ISA_use
@ CHECK-ATTR:     Description: Thumb-2
@ CHECK-ATTR:   }
@ CHECK-ATTR: }

