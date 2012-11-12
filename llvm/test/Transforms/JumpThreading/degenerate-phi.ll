; RUN: opt -jump-threading -disable-output %s
; PR9112

; This is actually a test for value tracking. Jump threading produces
; "%phi = phi i16" when it removes all edges leading to %unreachable.
; The .ll parser won't let us write that directly since it's invalid code.

define void @func() nounwind {
entry:
  br label %bb

bb:
  br label %bb

unreachable:
  %phi = phi i16 [ %add, %unreachable ], [ 0, %next ]
  %add = add i16 0, %phi
  %cmp = icmp slt i16 %phi, 0
  br i1 %cmp, label %unreachable, label %next

next:
  br label %unreachable
}

