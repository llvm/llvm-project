; REQUIRES: asserts
; RUN: opt < %s -passes=loop-interchange -S -debug 2>&1 | FileCheck %s

; CHECK:       Dependency matrix before interchange:
; CHECK-NEXT:  I I
; CHECK-NEXT:  = S
; CHECK-NEXT:  < S
; CHECK-NEXT:  Processing InnerLoopId

; This example is taken from github issue #54176
;
define void @foo(i32 noundef %n, i32 noundef %m, ptr nocapture noundef %aa, ptr nocapture noundef readonly %bb, ptr nocapture noundef writeonly %cc) {
entry:
  %arrayidx7 = getelementptr inbounds i8, ptr %aa, i64 512
  br label %for.cond1.preheader

for.cond1.preheader:
  %indvars.iv32 = phi i64 [ 1, %entry ], [ %indvars.iv.next33, %for.cond.cleanup3 ]
  %0 = add nsw i64 %indvars.iv32, -1
  %arrayidx9 = getelementptr inbounds [128 x float], ptr %arrayidx7, i64 0, i64 %0
  %arrayidx12 = getelementptr inbounds [128 x float], ptr %arrayidx7, i64 0, i64 %indvars.iv32
  br label %for.body4

for.cond.cleanup:
  ret void

for.cond.cleanup3:
  %indvars.iv.next33 = add nuw nsw i64 %indvars.iv32, 1
  %exitcond36 = icmp ne i64 %indvars.iv.next33, 128
  br i1 %exitcond36, label %for.cond1.preheader, label %for.cond.cleanup

for.body4:
  %indvars.iv = phi i64 [ 1, %for.cond1.preheader ], [ %indvars.iv.next, %for.body4 ]
  %arrayidx6 = getelementptr inbounds [128 x float], ptr %bb, i64 %indvars.iv, i64 %indvars.iv32
  %1 = load float, ptr %arrayidx6, align 4
  %2 = load float, ptr %arrayidx9, align 4
  %add = fadd fast float %2, %1
  store float %add, ptr %arrayidx9, align 4
  %3 = load float, ptr %arrayidx12, align 4
  %arrayidx16 = getelementptr inbounds [128 x float], ptr %cc, i64 %indvars.iv, i64 %indvars.iv32
  store float %3, ptr %arrayidx16, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.body4, label %for.cond.cleanup3
}
