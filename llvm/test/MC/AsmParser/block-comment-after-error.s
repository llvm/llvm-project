# RUN: not llvm-mc -triple x86_64 %s -o /dev/null 2>&1 \
# RUN:   | FileCheck %s

## The error path into eatToEndOfStatement().  Without the block comment skip
## in parseStatement() the comment-only line below is taken for the start of a
## statement and draws a second, spurious error.  See
## block-comment-at-statement-start.s for the error-free cases.
##
## The directives sit above the input on purpose: a # line comment between the
## two lines lexes as an EndOfStatement and resets the lexer, which hides the
## very thing this is testing.

# CHECK: [[#@LINE+2]]:{{[0-9]+}}: error: unexpected token
# CHECK-NOT: error: unexpected token at start of statement
	.byte 1 2 /* trailing */
/* a line that is nothing but a comment */
	nop
