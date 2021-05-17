@ RUN: not llvm-mc -triple armv7-eabi -o /dev/null 2>&1 %s | FileCheck %s

	.syntax unified

	.thumb

	.thumb_set

@ CHECK: error: expected identifier after '.thumb_set'
@ CHECK: 	.thumb_set
@ CHECK:                  ^

	.thumb_set ., 0x0b5e55ed

@ CHECK: error: expected identifier after '.thumb_set'
@ CHECK: 	.thumb_set ., 0x0b5e55ed
@ CHECK:                   ^

	.thumb_set labelled, 0x1abe11ed
	.thumb_set invalid, :lower16:labelled

@ CHECK: error: unknown token in expression
@ CHECK: 	.thumb_set invalid, :lower16:labelled
@ CHECK:                            ^

	.thumb_set missing_comma

@ CHECK: error: expected comma after name 'missing_comma'
@ CHECK: 	.thumb_set missing_comma
@ CHECK:                                ^

	.thumb_set missing_expression,

@ CHECK: error: missing expression
@ CHECK: 	.thumb_set missing_expression,
@ CHECK:                                      ^

	.thumb_set trailer_trash, 0x11fe1e55,

@ CHECK: error: expected newline
@ CHECK: 	.thumb_set trailer_trash, 0x11fe1e55,
@ CHECK:                                            ^

	.type alpha,%function
alpha:
	nop

        .type beta,%function
beta:
	bkpt

	.thumb_set beta, alpha

@ CHECK: error: redefinition of 'beta'
@ CHECK: 	.thumb_set beta, alpha
@ CHECK:                                            ^

	.type recursive_use,%function
	.thumb_set recursive_use, recursive_use + 1

@ CHECK: error: Recursive use of 'recursive_use'
@ CHECK: 	.thumb_set recursive_use, recursive_use + 1
@ CHECK:                                            ^

  variable_result = alpha + 1
  .long variable_result
	.thumb_set variable_result, 1

@ CHECK: error: invalid reassignment of non-absolute variable 'variable_result'
@ CHECK: 	.thumb_set variable_result, 1
@ CHECK:                                            ^
