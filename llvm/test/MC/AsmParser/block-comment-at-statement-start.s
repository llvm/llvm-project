# RUN: llvm-mc -triple x86_64 %s -o /dev/null 2>&1 \
# RUN:   | FileCheck %s --allow-empty --implicit-check-not={{.}}

## eatToEndOfStatement() lexes raw, so it can leave a block comment as the
## current token.  parseStatement() has to skip it, or each shape below is
## rejected with "unexpected token at start of statement" -- input GNU as
## accepts.  .extern reaches eatToEndOfStatement() with no error involved, so
## none of this depends on error recovery.

	.extern a
/* comment-only line */
	nop

	.extern b
/* spanning
   several
   lines */
	nop

	.extern c
/* two */ /* on one line */
	nop

	.extern d
/* followed by an instruction */ nop

## A block comment as the last thing in the file exercises the Lex()-at-EOF
## path in the skip loop.
	.extern e
/* at end of file */
