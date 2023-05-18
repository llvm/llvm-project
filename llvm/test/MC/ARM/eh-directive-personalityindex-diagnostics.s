@ RUN: not llvm-mc -triple armv7-linux-eabi -filetype asm -o /dev/null %s 2>&1  \
@ RUN:   | FileCheck %s

	.syntax unified
	.thumb

	.global function
	.type function,%function
	.thumb_func
function:
	.personalityindex 0

@ CHECK: error: .fnstart must precede .personalityindex directive
@ CHECK: 	.personalityindex 0
@ CHECK:        ^

	.global ununwindable
	.type ununwindable,%function
	.thumb_func
ununwindable:
	.fnstart
	.cantunwind
	.personalityindex 0
	.fnend

@ CHECK: error: .personalityindex cannot be used with .cantunwind
@ CHECK: 	.personalityindex 0
@ CHECK:        ^
@ CHECK: note: .cantunwind was specified here
@ CHECK: 	.cantunwind
@ CHECK:        ^

	.global nodata
	.type nodata,%function
	.thumb_func
nodata:
	.fnstart
	.handlerdata
	.personalityindex 0
	.fnend

@ CHECK: error: .personalityindex must precede .handlerdata directive
@ CHECK: 	.personalityindex 0
@ CHECK:        ^
@ CHECK: note: .handlerdata was specified here
@ CHECK: 	.handlerdata
@ CHECK:        ^

	.global multiple_personality
	.type multiple_personality,%function
	.thumb_func
multiple_personality:
	.fnstart
	.personality __aeabi_personality_pr0
	.personalityindex 0
	.fnend

@ CHECK: error: multiple personality directives
@ CHECK: 	.personalityindex 0
@ CHECK:        ^
@ CHECK: note: .personality was specified here
@ CHECK: 	.personality __aeabi_personality_pr0
@ CHECK:        ^
@ CHECK: note: .personalityindex was specified here
@ CHECK: 	.personalityindex 0
@ CHECK:       ^

	.global multiple_personality_indicies
	.type multiple_personality_indicies,%function
	.thumb_func
multiple_personality_indicies:
	.fnstart
	.personalityindex 0
	.personalityindex 1
	.fnend

@ CHECK: error: multiple personality directives
@ CHECK: 	.personalityindex 1
@ CHECK:        ^
@ CHECK: note: .personalityindex was specified here
@ CHECK: 	.personalityindex 0
@ CHECK:        ^
@ CHECK: note: .personalityindex was specified here
@ CHECK: 	.personalityindex 1
@ CHECK:        ^

	.global invalid_expression
	.type invalid_expression,%function
	.thumb_func
invalid_expression:
	.fnstart
	.personalityindex <expression>
	.fnend

@ CHECK: error: unknown token in expression
@ CHECK: 	.personalityindex <expression>
@ CHECK:                          ^

	.global nonconstant_expression
	.type nonconstant_expression,%function
	.thumb_func
nonconstant_expression:
	.fnstart
	.personalityindex nonconstant_expression
	.fnend

@ CHECK: error: index must be a constant number
@ CHECK: 	.personalityindex nonconstant_expression
@ CHECK:                          ^

	.global bad_index
	.type bad_index,%function
	.thumb_func
bad_index:
	.fnstart
	.personalityindex 42
	.fnend

@ CHECK: error: personality routine index should be in range [0-3]
@ CHECK: 	.personalityindex 42
@ CHECK:                          ^

