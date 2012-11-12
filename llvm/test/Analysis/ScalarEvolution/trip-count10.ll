; RUN: opt < %s -analyze -scalar-evolution | FileCheck %s

; Trip counts with trivial exit conditions.

; CHECK: Determining loop execution counts for: @a
; CHECK: Loop %loop: Unpredictable backedge-taken count. 
; CHECK: Loop %loop: Unpredictable max backedge-taken count.

; CHECK: Determining loop execution counts for: @b
; CHECK: Loop %loop: backedge-taken count is false
; CHECK: Loop %loop: max backedge-taken count is false

; CHECK: Determining loop execution counts for: @c
; CHECK: Loop %loop: backedge-taken count is false
; CHECK: Loop %loop: max backedge-taken count is false

; CHECK: Determining loop execution counts for: @d
; CHECK: Loop %loop: Unpredictable backedge-taken count. 
; CHECK: Loop %loop: Unpredictable max backedge-taken count. 

define void @a(i64 %n) nounwind {
entry:
  %t0 = icmp sgt i64 %n, 0
  br i1 %t0, label %loop, label %return

loop:
  %i = phi i64 [ %i.next, %loop ], [ 0, %entry ]
  %i.next = add nsw i64 %i, 1
  %exitcond = icmp eq i64 %i.next, %n
  br i1 false, label %return, label %loop

return:
  ret void
}
define void @b(i64 %n) nounwind {
entry:
  %t0 = icmp sgt i64 %n, 0
  br i1 %t0, label %loop, label %return

loop:
  %i = phi i64 [ %i.next, %loop ], [ 0, %entry ]
  %i.next = add nsw i64 %i, 1
  %exitcond = icmp eq i64 %i.next, %n
  br i1 true, label %return, label %loop

return:
  ret void
}
define void @c(i64 %n) nounwind {
entry:
  %t0 = icmp sgt i64 %n, 0
  br i1 %t0, label %loop, label %return

loop:
  %i = phi i64 [ %i.next, %loop ], [ 0, %entry ]
  %i.next = add nsw i64 %i, 1
  %exitcond = icmp eq i64 %i.next, %n
  br i1 false, label %loop, label %return

return:
  ret void
}
define void @d(i64 %n) nounwind {
entry:
  %t0 = icmp sgt i64 %n, 0
  br i1 %t0, label %loop, label %return

loop:
  %i = phi i64 [ %i.next, %loop ], [ 0, %entry ]
  %i.next = add nsw i64 %i, 1
  %exitcond = icmp eq i64 %i.next, %n
  br i1 true, label %loop, label %return

return:
  ret void
}

; Trip counts for non-polynomial iterations. It's theoretically possible
; to compute a maximum count for these, but short of that, ScalarEvolution
; should return unknown.

; PR7416
; CHECK: Determining loop execution counts for: @nonpolynomial
; CHECK-NEXT: Loop %loophead: Unpredictable backedge-taken count
; CHECK-NEXT: Loop %loophead: Unpredictable max backedge-taken count

declare i1 @g() nounwind

define void @nonpolynomial() {
entry:
  br label %loophead
loophead:
  %x = phi i32 [0, %entry], [%x.1, %bb1], [%x.2, %bb2]
  %y = icmp slt i32 %x, 100
  br i1 %y, label %loopbody, label %retbb
loopbody:
  %z = call i1 @g()
  br i1 %z, label %bb1, label %bb2
bb1:
  %x.1 = add i32 %x, 2
  br label %loophead
bb2:
  %x.2 = add i32 %x, 3
  br label %loophead
retbb:
  ret void
}

; PHI nodes with all constant operands.

; CHECK: Determining loop execution counts for: @constant_phi_operands
; CHECK: Loop %loop: backedge-taken count is 1
; CHECK: Loop %loop: max backedge-taken count is 1

define void @constant_phi_operands() nounwind {
entry:
  br label %loop

loop:
  %i = phi i64 [ 1, %loop ], [ 0, %entry ]
  %exitcond = icmp eq i64 %i, 1
  br i1 %exitcond, label %return, label %loop

return:
  ret void
}
