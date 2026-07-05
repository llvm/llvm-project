// RUN: llvm-mc -triple i686-elf -filetype asm -o - %s | FileCheck %s

	.type TYPE STT_FUNC
// CHECK: .type TYPE,@function

	.type comma_TYPE, STT_FUNC
// CHECK: .type comma_TYPE,@function

	.type at_TYPE, @STT_FUNC
// CHECK: .type at_TYPE,@function

	.type percent_TYPE, %STT_FUNC
// CHECK: .type percent_TYPE,@function

	.type string_TYPE, "STT_FUNC"
// CHECK: .type string_TYPE,@function

	.type type function
// CHECK: .type type,@function

	.type comma_type, function
// CHECK: .type comma_type,@function

	.type at_type, @function
// CHECK: .type at_type,@function

	.type percent_type, %function
// CHECK: .type percent_type,@function

	.type string_type, "function"
// CHECK: .type string_type,@function
