; RUN: opt < %s -loop-deletion -S | FileCheck %s

; Checks whether dead loops with multiple exits can be eliminated

; CHECK:      entry:
; CHECK-NEXT:   br label %return

; CHECK:      return:
; CHECK-NEXT:   ret void

define void @foo(i64 %n, i64 %m) nounwind {
entry:
  br label %bb

bb:
  %x.0 = phi i64 [ 0, %entry ], [ %t0, %bb2 ]
  %t0 = add i64 %x.0, 1
  %t1 = icmp slt i64 %x.0, %n
  br i1 %t1, label %bb2, label %return
bb2:
  %t2 = icmp slt i64 %x.0, %m
  br i1 %t1, label %bb, label %return

return:
  ret void
}
