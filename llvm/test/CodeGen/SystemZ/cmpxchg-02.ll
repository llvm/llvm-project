; Test 16-bit compare and swap.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s -check-prefix=CHECK-MAIN
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s -check-prefix=CHECK-SHIFT

; Check compare and swap with a variable.
; - CHECK is for the main loop.
; - CHECK-SHIFT makes sure that the negated shift count used by the second
;   RLL is set up correctly.  The negation is independent of the NILL and L
;   tested in CHECK.  CHECK-SHIFT also checks that %r3 is not modified before
;   being used in the RISBG (in contrast to things like atomic addition,
;   which shift %r3 left so that %b is at the high end of the word).
define i16 @f1(i16 %dummy, i16 *%src, i16 %cmp, i16 %swap) {
; CHECK-MAIN-LABEL: f1:
; CHECK-MAIN: risbg [[RISBG:%r[1-9]+]], %r3, 0, 189, 0{{$}}
; CHECK-MAIN: sll %r3, 3
; CHECK-MAIN: l [[OLD:%r[0-9]+]], 0([[RISBG]])
; CHECK-MAIN: [[LOOP:\.[^ ]*]]:
; CHECK-MAIN: rll %r2, [[OLD]], 16(%r3)
; CHECK-MAIN: risbg %r4, %r2, 32, 47, 0
; CHECK-MAIN: crjlh %r2, %r4, [[EXIT:\.[^ ]*]]
; CHECK-MAIN: risbg %r5, %r2, 32, 47, 0
; CHECK-MAIN: rll [[NEW:%r[0-9]+]], %r5, -16({{%r[1-9]+}})
; CHECK-MAIN: cs [[OLD]], [[NEW]], 0([[RISBG]])
; CHECK-MAIN: jl [[LOOP]]
; CHECK-MAIN: [[EXIT]]:
; CHECK-MAIN-NOT: %r2
; CHECK-MAIN: br %r14
;
; CHECK-SHIFT-LABEL: f1:
; CHECK-SHIFT: sll %r3, 3
; CHECK-SHIFT: lcr [[NEGSHIFT:%r[1-9]+]], %r3
; CHECK-SHIFT: rll
; CHECK-SHIFT: rll {{%r[0-9]+}}, %r5, -16([[NEGSHIFT]])
  %pair = cmpxchg i16 *%src, i16 %cmp, i16 %swap seq_cst seq_cst
  %res = extractvalue { i16, i1 } %pair, 0
  ret i16 %res
}

; Check compare and swap with constants.  We should force the constants into
; registers and use the sequence above.
define i16 @f2(i16 *%src) {
; CHECK-LABEL: f2:
; CHECK: lhi [[CMP:%r[0-9]+]], 42
; CHECK: risbg [[CMP]], {{%r[0-9]+}}, 32, 47, 0
; CHECK: risbg
; CHECK: br %r14
;
; CHECK-SHIFT-LABEL: f2:
; CHECK-SHIFT: lhi [[SWAP:%r[0-9]+]], 88
; CHECK-SHIFT: risbg
; CHECK-SHIFT: risbg [[SWAP]], {{%r[0-9]+}}, 32, 47, 0
; CHECK-SHIFT: br %r14
  %pair = cmpxchg i16 *%src, i16 42, i16 88 seq_cst seq_cst
  %res = extractvalue { i16, i1 } %pair, 0
  ret i16 %res
}
