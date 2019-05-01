@ RUN: llvm-mc -triple armv7-eabi -filetype obj -o - %s \
@ RUN:   | llvm-readobj --arm-attributes | FileCheck %s -check-prefix CHECK-ATTR

	.syntax unified
	.thumb
	.eabi_attribute Tag_compatibility, 1, "aeabi"

@ CHECK-ATTR: FileAttributes {
@ CHECK-ATTR:   Attribute {
@ CHECK-ATTR:     Value: 1, aeabi
@ CHECK-ATTR:     TagName: compatibility
@ CHECK-ATTR:     Description: AEABI Conformant
@ CHECK-ATTR:   }
@ CHECK-ATTR: }

