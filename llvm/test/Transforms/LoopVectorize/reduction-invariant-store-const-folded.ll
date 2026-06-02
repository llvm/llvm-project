; RUN: opt < %s -passes="loop-vectorize" -S | FileCheck %s
;
; Verify that loop-vectorize does not crash when a reduction with an invariant
; store is simplified to a constant (or x, -1 -> -1).

define void @reduction_store_const_folded() {
; CHECK-LABEL: define void @reduction_store_const_folded()
; CHECK-NOT: vector.body
; CHECK: ret void
entry:
  br label %loop

loop:
  %j = phi i32 [ 0, %entry ], [ %add, %loop ]
  %or810 = phi i32 [ 0, %entry ], [ %or, %loop ]
  %or = or i32 %or810, -1
  store i32 %or, ptr null, align 4
  %add = add i32 %j, 1
  %cmp = icmp slt i32 %j, 1
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}
