@ RUN: not llvm-mc -triple armv7-linux-eabi -filetype asm -o /dev/null 2>&1 %s \
@ RUN:   | FileCheck %s

	.syntax unified

	.type require_fnstart,%function
require_fnstart:
	.unwind_raw 0, 0

@ CHECK: error: .fnstart must precede .unwind_raw directive
@ CHECK: 	.unwind_raw 0, 0
@ CHECK:        ^

	.type check_arguments,%function
check_arguments:
	.fnstart
	.unwind_raw
	.fnend

@ CHECK: error: expected expression
@ CHECK: 	.unwind_raw
@ CHECK:                   ^

	.type check_stack_offset,%function
check_stack_offset:
	.fnstart
	.unwind_raw ., 0
	.fnend

@ CHECK: error: offset must be a constant
@ CHECK: 	.unwind_raw ., 0
@ CHECK:                    ^

	.type comma_check,%function
comma_check:
	.fnstart
	.unwind_raw 0
	.fnend

@ CHECK: error: expected comma
@ CHECK: 	.unwind_raw 0
@ CHECK:                     ^

	.type require_opcode,%function
require_opcode:
	.fnstart
	.unwind_raw 0,
	.fnend

@ CHECK: error: expected opcode expression
@ CHECK: 	.unwind_raw 0,
@ CHECK:                      ^

	.type require_opcode_constant,%function
require_opcode_constant:
	.fnstart
	.unwind_raw 0, .
	.fnend

@ CHECK: error: opcode value must be a constant
@ CHECK: 	.unwind_raw 0, .
@ CHECK:                       ^

	.type check_opcode_range,%function
check_opcode_range:
	.fnstart
	.unwind_raw 0, 0x100
	.fnend

@ CHECK: error: invalid opcode
@ CHECK: 	.unwind_raw 0, 0x100
@ CHECK:                       ^

