; RUN: llc -O1 -mtriple=hexagon < %s | FileCheck %s

; A counted unsigned less-than loop whose start is a register value gets a
; distance guard. The guard must compare the distance as unsigned, otherwise a
; distance with the sign bit set is treated as negative and the trip count is
; clamped to 1.

; CHECK-LABEL: f:
; CHECK:       p0 = cmp.gtu(r{{[0-9]+}},#0)
; CHECK:       loop0(
; CHECK-NOT:   cmp.gt(r

define void @f(ptr %p, i32 %start, i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ %start, %entry ], [ %next, %loop ]
  %addr = getelementptr i8, ptr %p, i32 %i
  store i8 0, ptr %addr
  %next = add i32 %i, 1
  %cond = icmp ult i32 %next, %n
  br i1 %cond, label %loop, label %exit

exit:
  ret void
}
