@ Test the .arch directive for armv8.1-m.main

@ This test case will check the default .ARM.attributes value for the
@ armv8.1-m.main architecture when using the armv8.1m.main alias.

@ RUN: llvm-mc -triple arm-eabi -filetype asm %s \
@ RUN:   | FileCheck %s -check-prefix CHECK-ASM
@ RUN: llvm-mc -triple arm-eabi -filetype obj %s \
@ RUN:   | llvm-readobj --arch-specific - | FileCheck %s -check-prefix CHECK-ATTR

	.syntax	unified
	.arch	armv8.1m.main

@ CHECK-ASM: 	.arch	armv8.1-m.main

@ CHECK-ATTR: FileAttributes {
@ CHECK-ATTR:   Attribute {
@ CHECK-ATTR:     TagName: CPU_name
@ CHECK-ATTR:     Value: 8.1-M.Mainline
@ CHECK-ATTR:   }
@ CHECK-ATTR:   Attribute {
@ CHECK-ATTR:     TagName: CPU_arch
@ CHECK-ATTR:     Description: ARM v8.1-M Mainline
@ CHECK-ATTR:   }
@ CHECK-ATTR:   Attribute {
@ CHECK-ATTR:     TagName: CPU_arch_profile
@ CHECK-ATTR:     Description: Microcontroller
@ CHECK-ATTR:   }
@ CHECK-ATTR:   Attribute {
@ CHECK-ATTR:     TagName: THUMB_ISA_use
@ CHECK-ATTR:     Description: Permitted
@ CHECK-ATTR:   }
@ CHECK-ATTR: }
