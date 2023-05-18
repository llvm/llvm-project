@ RUN: not llvm-mc -triple armv7-elf -filetype asm -o /dev/null %s 2>&1 \
@ RUN:   | FileCheck %s

	.syntax unified
	.thumb

	.eabi_attribute Tag_unknown_name, 0
@ CHECK: error: attribute name not recognised: Tag_unknown_name
@ CHECK: 	.eabi_attribute Tag_unknown_name
@ CHECK:                        ^

	.eabi_attribute [non_constant_expression], 0
@ CHECK: error: expected numeric constant
@ CHECK: 	.eabi_attribute [non_constant_expression], 0
@ CHECK:                        ^

	.eabi_attribute 42, "forty two"
@ CHECK: error: expected numeric constant
@ CHECK: 	.eabi_attribute 42, "forty two"
@ CHECK:                            ^

	.eabi_attribute 43, 43
@ CHECK: error: bad string constant
@ CHECK: 	.eabi_attribute 43, 43
@ CHECK:                            ^

	.eabi_attribute 0
@ CHECK: :[[#@LINE-1]]:19: error: expected comma
@ CHECK: 	.eabi_attribute 0
@ CHECK:                         ^

        .eabi_attribute Tag_compatibility, 1
@ CHECK: :[[#@LINE-1]]:45: error: expected comma
@ CHECK: .eabi_attribute Tag_compatibility, 1
@ CHECK:                                     ^

	.eabi_attribute Tag_MPextension_use_old, 0
@ CHECK: error: attribute name not recognised: Tag_MPextension_use_old
@ CHECK: 	.eabi_attribute Tag_MPextension_use_old, 0
@ CHECK:                        ^

