# RUN: llvm-mc -triple x86_64 -preserve-comments %s \
# RUN:   | FileCheck %s --strict-whitespace

## parseStatement() skips a block comment left as the current token by
## eatToEndOfStatement().  It has to hand that comment to the streamer on the
## way past, or -preserve-comments drops it -- unlike a block comment the
## ordinary Lex() path sees.  See block-comment-at-statement-start.s for why
## the comment lands there in the first place.
##
## The {{^}} anchors keep these directives from matching their own echo, which
## -preserve-comments emits as well.

	.extern a
/* preserved */
	nop
# CHECK: {{^}}	# preserved

	.extern b
/* spanning
   two lines */
	nop
# CHECK: {{^}}	# spanning
# CHECK: {{^}}	#   two lines
