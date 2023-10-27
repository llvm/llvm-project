; RUN: opt < %s -S -passes=loop-unroll -unroll-runtime=true -unroll-count=4 | FileCheck %s

;; Check that the remainder loop is properly assigned a branch weight for its latch branch.
; CHECK-LABEL: @test(
; CHECK-LABEL: for.body:
; CHECK: br i1 [[COND1:%.*]], label %for.end.loopexit.unr-lcssa.loopexit, label %for.body, !prof ![[#PROF:]], !llvm.loop ![[#LOOP:]]
; CHECK-LABEL: for.body.epil:
; CHECK: br i1 [[COND2:%.*]], label  %for.body.epil, label %for.end.loopexit.epilog-lcssa, !prof ![[#PROF2:]], !llvm.loop ![[#LOOP2:]]
; CHECK: ![[#PROF]] = !{!"branch_weights", i32 1, i32 2499}
; CHECK: ![[#PROF2]] = !{!"branch_weights", i32 1, i32 1}

define i3 @test(ptr %a, i3 %n) {
entry:
  %cmp1 = icmp eq i3 %n, 0
  br i1 %cmp1, label %for.end, label %for.body

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %sum.02 = phi i3 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i3, ptr %a, i64 %indvars.iv
  %0 = load i3, ptr %arrayidx
  %add = add nsw i3 %0, %sum.02
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i3
  %exitcond = icmp eq i3 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body, !prof !0

for.end:
  %sum.0.lcssa = phi i3 [ 0, %entry ], [ %add, %for.body ]
  ret i3 %sum.0.lcssa
}

!0 = !{!"branch_weights", i32 1, i32 9999}
