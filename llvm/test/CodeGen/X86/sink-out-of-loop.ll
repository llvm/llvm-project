; RUN: llc -mtriple=x86_64-apple-darwin < %s | FileCheck %s

; A MOV32ri is inside a loop, it has two successors, one successor is inside the
; same loop, the other successor is outside the loop. We should be able to sink
; MOV32ri outside the loop.
; rdar://11980766
define i32 @sink_succ(i32 %argc, i8** nocapture %argv) nounwind uwtable ssp {
; CHECK: sink_succ
; CHECK: [[OUTER_LN1:LBB0_[0-9]+]]: ## %preheader
; CHECK: %exit
; CHECK-NOT: movl
; CHECK: jne [[OUTER_LN1]]
; CHECK: movl
; CHECK: [[LN2:LBB0_[0-9]+]]: ## %for.body2
; CHECK: jne [[LN2]]
; CHECK: ret
entry:
  br label %preheader

preheader:
  %i.127 = phi i32 [ 0, %entry ], [ %inc9, %exit ]
  br label %for.body1.lr

for.body1.lr:
  %iv30 = phi i32 [ 1, %preheader ], [ %iv.next31, %for.inc40.i ]
  br label %for.body1

for.body1:
  %iv.i = phi i64 [ 0, %for.body1.lr ], [ %iv.next.i, %for.body1 ]
  %iv.next.i = add i64 %iv.i, 1
  %lftr.wideiv32 = trunc i64 %iv.next.i to i32
  %exitcond33 = icmp eq i32 %lftr.wideiv32, %iv30
  br i1 %exitcond33, label %for.inc40.i, label %for.body1

for.inc40.i:
  %iv.next31 = add i32 %iv30, 1
  %exitcond49.i = icmp eq i32 %iv.next31, 32
  br i1 %exitcond49.i, label %exit, label %for.body1.lr

exit:
  %inc9 = add nsw i32 %i.127, 1
  %exitcond34 = icmp eq i32 %inc9, 10
  br i1 %exitcond34, label %for.body2, label %preheader

for.body2:
  %iv = phi i64 [ %iv.next, %for.body2 ], [ 0, %exit ]
  %iv.next = add i64 %iv, 1
  %lftr.wideiv = trunc i64 %iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 2048
  br i1 %exitcond, label %for.end20, label %for.body2

for.end20:
  ret i32 0
}
