@ RUN: not llvm-mc -triple armv7-eabi -filetype asm -o /dev/null 2>&1 %s \
@ RUN:  | FileCheck %s

	.syntax unified
	.thumb

	.global false_start
	.type false_start,%function
	.thumb_func
false_start:
	.movsp r7

@ CHECK: error: .fnstart must precede .movsp directive
@ CHECK: 	.movsp r7
@ CHECK:        ^

	.global beyond_saving
	.type beyond_saving,%function
	.thumb_func
beyond_saving:
	.fnstart
	.setfp r11, sp, #8
	add r11, sp, #8
	.movsp r7
	mov r7, r11
	.fnend

@ CHECK: error: unexpected .movsp directive
@ CHECK: 	.movsp r7
@ CHECK:        ^


	.global sp_invalid
	.type sp_invalid,%function
	.thumb_func
sp_invalid:
	.fnstart
	.movsp r13
	mov sp, sp
	.fnend

@ CHECK: error: sp and pc are not permitted in .movsp directive
@ CHECK: 	.movsp r13
@ CHECK:               ^


	.global pc_invalid
	.type pc_invalid,%function
	.thumb_func
pc_invalid:
	.fnstart
	.movsp r15
	mov sp, pc
	.fnend

@ CHECK: error: sp and pc are not permitted in .movsp directive
@ CHECK: 	.movsp r15
@ CHECK:               ^


	.global constant_required
	.type constant_required,%function
	.thumb_func
constant_required:
	.fnstart
	.movsp r11,
	mov sp, r11
	.fnend

@ CHECK: error: expected #constant
@ CHECK: 	.movsp r11,
@ CHECK:                   ^


	.global constant_constant
	.type constant_constant,%function
	.thumb_func
constant_constant:
	.fnstart
	.movsp r11, #constant
	mov sp, r11
	.fnend

@ CHECK: error: offset must be an immediate constant
@ CHECK: 	.movsp r11, #constant
@ CHECK:                     ^


	.arm

	.global register_required
	.type register_required,%function
register_required:
	.fnstart
	.movsp #42
	mov sp, #42
	.fnend

@ CHECK: error: register expected
@ CHECK: 	.movsp #42
@ CHECK:               ^

