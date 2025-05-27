@ RUN: not llvm-mc -triple armv7-eabi -o /dev/null %s 2>&1 | FileCheck %s
@ RUN: not llvm-mc -filetype=obj -triple=armv7-eabi -o /dev/null %s --defsym LOOP=1 2>&1 | FileCheck %s --check-prefix=ERR2 --implicit-check-not=error:

	.syntax unified

	.thumb

.ifndef LOOP
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
@ CHECK: :[[#@LINE-1]]:26: error: expected comma
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

  variable_result = alpha + 1
  .long variable_result
	.thumb_set variable_result, 1

@ CHECK: error: invalid reassignment of non-absolute variable 'variable_result'
@ CHECK: 	.thumb_set variable_result, 1
@ CHECK:                                            ^

.else
.type recursive_use,%function
.thumb_set recursive_use, recursive_use + 1
@ ERR2: [[#@LINE-1]]:41: error: cyclic dependency detected for symbol 'recursive_use'
@ ERR2: [[#@LINE-2]]:41: error: expression could not be evaluated

.type recursive_use2,%function
.thumb_set recursive_use2, recursive_use2 + 1
@ ERR2: [[#@LINE-1]]:43: error: cyclic dependency detected for symbol 'recursive_use2'
@ ERR2: [[#@LINE-2]]:43: error: expression could not be evaluated
.endif
