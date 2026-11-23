## A block comment after a statement discarded by eatToEndOfStatement() is not the start of the next statement.
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -triple=x86_64 -preserve-comments a.s | FileCheck %s --match-full-lines --strict-whitespace
# RUN: llvm-mc -triple=x86_64 include.s | FileCheck %s --check-prefix=INC --match-full-lines --strict-whitespace
# RUN: not llvm-mc -triple=x86_64 err.s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR --implicit-check-not=error:

#      CHECK:	# comment{{ }}
# CHECK-NEXT:	nop
#CHECK-EMPTY:
# CHECK-NEXT:	# space+comment{{ }}
# CHECK-NEXT:	nop
#CHECK-EMPTY:
# CHECK-NEXT:	# spanning
# CHECK-NEXT:	#  two lines{{ }}
# CHECK-NEXT:	nop
#CHECK-EMPTY:
# CHECK-NEXT:	nop	# then an instruction{{ }}
#CHECK-EMPTY:
# CHECK-NEXT:	nop	# line comment
#CHECK-EMPTY:
# CHECK-NEXT:	# at end of file{{ }}

#       INC:	retq
# INC-EMPTY:
#       INC:	nop

#--- a.s
.extern a
/* comment */
nop

.extern a1
  /* space+comment */
nop

.extern b
/* spanning
  two lines */
nop

.extern c
/* then an instruction */ nop

.extern d # line comment
nop

.extern e
/* at end of file */

#--- include.s
.include "included.s"
nop

#--- included.s
ret
.extern b
/* comment */
.extern a
#--- err.s
.byte 1 2
/* comment-only line */
nop

#      ERR:err.s:1:9: error: unexpected token
# ERR-NEXT:.byte 1 2
