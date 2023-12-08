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
define i16 @f1(i16 %dummy, ptr %src, i16 %cmp, i16 %swap) {
; CHECK-MAIN-LABEL: f1:
; CHECK-MAIN: risbg [[RISBG:%r[1-9]+]], %r3, 0, 189, 0{{$}}
; CHECK-MAIN-DAG: sll %r3, 3
; CHECK-MAIN-DAG: l [[OLD:%r[0-9]+]], 0([[RISBG]])
; CHECK-MAIN-DAG: llhr %r4, %r4
; CHECK-MAIN: [[LOOP:\.[^ ]*]]:
; CHECK-MAIN: rll %r2, [[OLD]], 16(%r3)
; CHECK-MAIN: risbg %r5, %r2, 32, 47, 0
; CHECK-MAIN: llhr %r2, %r2
; CHECK-MAIN: crjlh %r2, %r4, [[EXIT:\.[^ ]*]]
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
  %pair = cmpxchg ptr %src, i16 %cmp, i16 %swap seq_cst seq_cst
  %res = extractvalue { i16, i1 } %pair, 0
  ret i16 %res
}

; Check compare and swap with constants.  We should force the constants into
; registers and use the sequence above.
define i16 @f2(ptr %src) {
; CHECK-LABEL: f2:
; CHECK: lhi [[CMP:%r[0-9]+]], 42
; CHECK: risbg [[CMP]], {{%r[0-9]+}}, 32, 47, 0
; CHECK: risbg
; CHECK: br %r14
;
; CHECK-SHIFT-LABEL: f2:
; CHECK-SHIFT: lhi [[SWAP:%r[0-9]+]], 88
; CHECK-SHIFT: risbg [[SWAP]], {{%r[0-9]+}}, 32, 47, 0
; CHECK-SHIFT: br %r14
  %pair = cmpxchg ptr %src, i16 42, i16 88 seq_cst seq_cst
  %res = extractvalue { i16, i1 } %pair, 0
  ret i16 %res
}

; Check generating the comparison result.
define i32 @f3(i16 %dummy, ptr %src, i16 %cmp, i16 %swap) {
; CHECK-MAIN-LABEL: f3:
; CHECK-MAIN: risbg [[RISBG:%r[1-9]+]], %r3, 0, 189, 0{{$}}
; CHECK-MAIN-DAG: sll %r3, 3
; CHECK-MAIN-DAG: l [[OLD:%r[0-9]+]], 0([[RISBG]])
; CHECK-MAIN-DAG: llhr %r2, %r4
; CHECK-MAIN: [[LOOP:\.[^ ]*]]:
; CHECK-MAIN: rll [[TMP:%r[0-9]+]], [[OLD]], 16(%r3)
; CHECK-MAIN: risbg %r5, [[TMP]], 32, 47, 0
; CHECK-MAIN: llhr %r14, %r14
; CHECK-MAIN: cr [[TMP]], %r2
; CHECK-MAIN: jlh [[EXIT:\.[^ ]*]]
; CHECK-MAIN: rll [[NEW:%r[0-9]+]], %r5, -16({{%r[1-9]+}})
; CHECK-MAIN: cs [[OLD]], [[NEW]], 0([[RISBG]])
; CHECK-MAIN: jl [[LOOP]]
; CHECK-MAIN: [[EXIT]]:
; CHECK-MAIN-NEXT: ipm %r2
; CHECK-MAIN-NEXT: afi %r2, -268435456
; CHECK-MAIN-NEXT: srl %r2, 31
; CHECK-MAIN-NOT: %r2
; CHECK-MAIN: br %r14
;
; CHECK-SHIFT-LABEL: f3:
; CHECK-SHIFT: sll %r3, 3
; CHECK-SHIFT: lcr [[NEGSHIFT:%r[1-9]+]], %r3
; CHECK-SHIFT: rll
; CHECK-SHIFT: rll {{%r[0-9]+}}, %r5, -16([[NEGSHIFT]])
  %pair = cmpxchg ptr %src, i16 %cmp, i16 %swap seq_cst seq_cst
  %val = extractvalue { i16, i1 } %pair, 1
  %res = zext i1 %val to i32
  ret i32 %res
}

declare void @g()

; Check using the comparison result for a branch.
; CHECK-LABEL: f4
; CHECK-MAIN-LABEL: f4:
; CHECK-MAIN: risbg [[RISBG:%r[1-9]+]], %r2, 0, 189, 0{{$}}
; CHECK-MAIN-DAG: sll %r2, 3
; CHECK-MAIN-DAG: l [[OLD:%r[0-9]+]], 0([[RISBG]])
; CHECK-MAIN-DAG: llhr %r3, %r3
; CHECK-MAIN: [[LOOP:\.[^ ]*]]:
; CHECK-MAIN: rll [[TMP:%r[0-9]+]], [[OLD]], 16(%r2)
; CHECK-MAIN: risbg %r4, [[TMP]], 32, 47, 0
; CHECK-MAIN: llhr %r14, %r14
; CHECK-MAIN: cr [[TMP]], %r3
; CHECK-MAIN: jlh [[EXIT:\.[^ ]*]]
; CHECK-MAIN: rll [[NEW:%r[0-9]+]], %r4, -16({{%r[1-9]+}})
; CHECK-MAIN: cs [[OLD]], [[NEW]], 0([[RISBG]])
; CHECK-MAIN: jl [[LOOP]]
; CHECK-MAIN: [[EXIT]]:
; CHECK-MAIN-NEXT: jlh [[LABEL:\.[^ ]*]]
; CHECK-MAIN: jg g
; CHECK-MAIN: [[LABEL]]:
; CHECK-MAIN: br %r14
;
; CHECK-SHIFT-LABEL: f4:
; CHECK-SHIFT: sll %r2, 3
; CHECK-SHIFT: lcr [[NEGSHIFT:%r[1-9]+]], %r2
; CHECK-SHIFT: rll
; CHECK-SHIFT: rll {{%r[0-9]+}}, %r4, -16([[NEGSHIFT]])
define void @f4(ptr %src, i16 %cmp, i16 %swap) {
  %pair = cmpxchg ptr %src, i16 %cmp, i16 %swap seq_cst seq_cst
  %cond = extractvalue { i16, i1 } %pair, 1
  br i1 %cond, label %call, label %exit

call:
  tail call void @g()
  br label %exit

exit:
  ret void
}

; ... and the same with the inverted direction.
; CHECK-MAIN-LABEL: f5:
; CHECK-MAIN: risbg [[RISBG:%r[1-9]+]], %r2, 0, 189, 0{{$}}
; CHECK-MAIN-DAG: sll %r2, 3
; CHECK-MAIN-DAG: l [[OLD:%r[0-9]+]], 0([[RISBG]])
; CHECK-MAIN-DAG: llhr %r3, %r3
; CHECK-MAIN: [[LOOP:\.[^ ]*]]:
; CHECK-MAIN: rll [[TMP:%r[0-9]+]], [[OLD]], 16(%r2)
; CHECK-MAIN: risbg %r4, [[TMP]], 32, 47, 0
; CHECK-MAIN: llhr %r14, %r14
; CHECK-MAIN: cr [[TMP]], %r3
; CHECK-MAIN: jlh [[EXIT:\.[^ ]*]]
; CHECK-MAIN: rll [[NEW:%r[0-9]+]], %r4, -16({{%r[1-9]+}})
; CHECK-MAIN: cs [[OLD]], [[NEW]], 0([[RISBG]])
; CHECK-MAIN: jl [[LOOP]]
; CHECK-MAIN: [[EXIT]]:
; CHECK-MAIN-NEXT: jlh [[LABEL:\.[^ ]*]]
; CHECK-MAIN: br %r14
; CHECK-MAIN: [[LABEL]]:
; CHECK-MAIN: jg g
;
; CHECK-SHIFT-LABEL: f5:
; CHECK-SHIFT: sll %r2, 3
; CHECK-SHIFT: lcr [[NEGSHIFT:%r[1-9]+]], %r2
; CHECK-SHIFT: rll
; CHECK-SHIFT: rll {{%r[0-9]+}}, %r4, -16([[NEGSHIFT]])
define void @f5(ptr %src, i16 %cmp, i16 %swap) {
  %pair = cmpxchg ptr %src, i16 %cmp, i16 %swap seq_cst seq_cst
  %cond = extractvalue { i16, i1 } %pair, 1
  br i1 %cond, label %exit, label %call

call:
  tail call void @g()
  br label %exit

exit:
  ret void
}

