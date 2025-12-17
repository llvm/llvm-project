@ RUN: rm -rf %t && split-file %s %t --leading-lines && cd %t
@ RUN: not llvm-mc -triple armv7-eabi -o /dev/null a.s 2>&1 | FileCheck %s
@ RUN: not llvm-mc -filetype=obj -triple=armv7-eabi -o /dev/null redef.s 2>&1 | FileCheck %s --check-prefix=ERR
@ RUN: not llvm-mc -filetype=obj -triple=armv7-eabi -o /dev/null cycle.s 2>&1 | FileCheck %s --check-prefix=ERR2 --implicit-check-not=error:

//--- a.s
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

//--- redef.s
	.type alpha,%function
alpha:
	nop
        .type beta,%function
beta:
.thumb_set beta, alpha
@ ERR: [[#@LINE-1]]:18: error: redefinition of 'beta'

  variable_result = alpha + 1
  .long variable_result
	.thumb_set variable_result, 1

//--- cycle.s
.type recursive_use,%function
.thumb_set recursive_use, recursive_use + 1
@ ERR2: [[#@LINE-1]]:41: error: cyclic dependency detected for symbol 'recursive_use'
@ ERR2: [[#@LINE-2]]:41: error: expression could not be evaluated

.type recursive_use2,%function
.thumb_set recursive_use2, recursive_use2 + 1
@ ERR2: [[#@LINE-1]]:43: error: cyclic dependency detected for symbol 'recursive_use2'
@ ERR2: [[#@LINE-2]]:43: error: expression could not be evaluated
