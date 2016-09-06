; RUN: opt < %s -loop-deletion -S | FileCheck %s
; RUN: opt < %s -passes='require<scalar-evolution>,loop(loop-deletion)' -S | FileCheck %s

; ScalarEvolution can prove the loop iteration is finite, even though
; it can't represent the exact trip count as an expression. That's
; good enough to let the loop be deleted.

; CHECK:      entry:
; CHECK-NEXT:   br label %return

; CHECK:      return:
; CHECK-NEXT:   ret void

define void @foo(i64 %n, i64 %m) nounwind {
entry:
  br label %bb

bb:
  %x.0 = phi i64 [ 0, %entry ], [ %t0, %bb ]
  %t0 = add i64 %x.0, 1
  %t1 = icmp slt i64 %x.0, %n
  %t3 = icmp sgt i64 %x.0, %m
  %t4 = and i1 %t1, %t3
  br i1 %t4, label %bb, label %return

return:
  ret void
}
