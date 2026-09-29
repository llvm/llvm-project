@ RUN: llvm-mc -triple arm-elf -filetype asm -o - %s | FileCheck %s

	.syntax unified

	.type TYPE #STT_FUNC
// CHECK: .type TYPE,%function

	.type type #function
// CHECK: .type type,%function

	.type comma_TYPE, #STT_FUNC
// CHECK: .type comma_TYPE,%function

	.type comma_type, #function
// CHECK: .type comma_type,%function

