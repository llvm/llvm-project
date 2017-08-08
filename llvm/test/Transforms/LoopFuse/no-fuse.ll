; RUN: opt -loop-fuse -verify-loop-info -verify-dom-info %s -S -o - | FileCheck %s

; 'C' equivalent: Partially generated and hand modified.
; void noFuse(int *a, int *b, int *c) {
;   for (i = 0; i < 1000; ++i)  // L1
;     c[i] = a[i] + c[i - 1];
;   for (i = 0; i < 1000; ++i)  // L2
;     c[i] = a[i] + b[i];
; }
; There is a backward dependence from L1 to L2. So it is unsafe to fuse.

; CHECK: entry:
; CHECK-NEXT:  br label %for.body
; CHECK: for.body:                                         ; preds = %for.body, %entry
; CHECK: for.body.1:

; ModuleID = '1.bc'

; Function Attrs: norecurse nounwind uwtable
define void @bigLoop(i32* nocapture readonly %a, i32* nocapture readonly %b, i32* nocapture %c) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %indvars.iv.next.back = add i64 %indvars.iv, -1
  %arrayidx3 = getelementptr inbounds i32, i32* %c, i64 %indvars.iv.next.back
  %1 = load i32, i32* %arrayidx3, align 4
  %add4 = add nsw i32 %1, %0
  %arrayidx6 = getelementptr inbounds i32, i32* %c, i64 %indvars.iv
  store i32 %add4, i32* %arrayidx6, align 4
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end.loopexit, label %for.body, !llvm.loop !4

for.end.loopexit:                                 ; preds = %for.body
  br label %for.body.1

for.body.1:                                       ; preds = %for.body.1, %for.end.loopexit
  %indvars.iv.1 = phi i64 [ 0, %for.end.loopexit ], [ %indvars.iv.next.1, %for.body.1 ]
  %arrayidx.1 = getelementptr inbounds i32, i32* %a, i64 %indvars.iv.1
  %2 = load i32, i32* %arrayidx.1, align 4
  %arrayidx10 = getelementptr inbounds i32, i32* %b, i64 %indvars.iv.1
  %3 = load i32, i32* %arrayidx10, align 4
  %add11 = add nsw i32 %3, %2
  %arrayidx12 = getelementptr inbounds i32, i32* %c, i64 %indvars.iv.1
  store i32 %add11, i32* %arrayidx12, align 4
  %indvars.iv.next.1 = add i64 %indvars.iv.1, 1
  %exitcond.1 = icmp eq i64 %indvars.iv.next.1, 1000
  br i1 %exitcond.1, label %for.end.loopexit.1, label %for.body.1, !llvm.loop !4

for.end.loopexit.1:                               ; preds = %for.body.1
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit.1
  ret void
}

attributes #0 = { norecurse nounwind uwtable }
attributes #1 = { norecurse nounwind readonly uwtable }
attributes #2 = { nounwind uwtable }
attributes #3 = { nounwind readonly }
attributes #4 = { nounwind }
attributes #5 = { noreturn nounwind }
attributes #6 = { nounwind readonly }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.8.0"}
!1 = distinct !{!1, !2, !3}
!2 = !{!"llvm.loop.vectorize.width", i32 1}
!3 = !{!"llvm.loop.interleave.count", i32 1}
!4 = distinct !{!4, !5}
!5 = !{!"llvm.loop.unroll.disable"}
!6 = distinct !{!6, !2, !3}
!7 = distinct !{!7, !2, !3}
