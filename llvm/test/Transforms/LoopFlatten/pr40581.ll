; RUN: opt < %s -S -passes='loop(loop-flatten),verify' -verify-loop-info -verify-dom-info -verify-scev | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

; Test case and IR corresponding to this code:
;
; int k = 0;
; for(int i = 0; i < n; i++)
;   for(int j = 0; j < n; j++) {
;     A[k] = B[k];
;     k++;
;   }
;
; TODO: this case doesn't trigger yet.
;
define dso_local void @v0(i32 %n, ptr nocapture %A, ptr nocapture readonly %B) local_unnamed_addr #0 {
;
; CHECK-LABEL: @v0
; CHECK-NOT:   %flatten.tripcount = mul i32 %n, %n
;
entry:
  %cmp21 = icmp sgt i32 %n, 0
  br i1 %cmp21, label %for.cond1.preheader.us.preheader, label %for.cond.cleanup

for.cond1.preheader.us.preheader:
  br label %for.cond1.preheader.us

for.cond1.preheader.us:
  %i.023.us = phi i32 [ %inc8.us, %for.cond1.for.cond.cleanup3_crit_edge.us ], [ 0, %for.cond1.preheader.us.preheader ]
  %k.022.us = phi i32 [ %inc.us.lcssa, %for.cond1.for.cond.cleanup3_crit_edge.us ], [ 0, %for.cond1.preheader.us.preheader ]
  %0 = add i32 %n, %k.022.us
  br label %for.body4.us

for.body4.us:
  %k.119.us = phi i32 [ %k.022.us, %for.cond1.preheader.us ], [ %inc.us, %for.body4.us ]
  %arrayidx.us = getelementptr inbounds i32, ptr %B, i32 %k.119.us
  %1 = load i32, ptr %arrayidx.us, align 4
  %arrayidx5.us = getelementptr inbounds i32, ptr %A, i32 %k.119.us
  store i32 %1, ptr %arrayidx5.us, align 4
  %inc.us = add i32 %k.119.us, 1
  %exitcond = icmp ne i32 %inc.us, %0
  br i1 %exitcond, label %for.body4.us, label %for.cond1.for.cond.cleanup3_crit_edge.us

for.cond1.for.cond.cleanup3_crit_edge.us:
  %inc.us.lcssa = phi i32 [ %inc.us, %for.body4.us ]
  %inc8.us = add nuw nsw i32 %i.023.us, 1
  %cmp.us = icmp slt i32 %inc8.us, %n
  br i1 %cmp.us, label %for.cond1.preheader.us, label %for.cond.cleanup.loopexit

for.cond.cleanup.loopexit:
  br label %for.cond.cleanup

for.cond.cleanup:
  ret void
}

; Test case and IR corresponding to this code:
;
; for(int i = 0; i < n; i++)
;   for(int j = 0; j < n; j++) {
;     int k = i*n+j;
;     A[k] = B[k];
;     k++;
;   }
;
define dso_local void @v1(i32 %n, ptr nocapture %A, ptr nocapture readonly %B) local_unnamed_addr #0 {
;
; CHECK-LABEL: @v1
; CHECK:       for.cond1.preheader.us.preheader:
; CHECK:         %flatten.tripcount = mul i32 %n, %n
; CHECK:       for.cond1.for.cond.cleanup3_crit_edge.us:
; CHECK:         %inc8.us = add nuw nsw i32 %i.024.us, 1
; CHECK:         %cmp.us = icmp slt i32 %inc8.us, %flatten.tripcount
;
entry:
  %cmp23 = icmp sgt i32 %n, 0
  br i1 %cmp23, label %for.cond1.preheader.us.preheader, label %for.cond.cleanup

for.cond1.preheader.us.preheader:
  br label %for.cond1.preheader.us

for.cond1.preheader.us:
  %i.024.us = phi i32 [ %inc8.us, %for.cond1.for.cond.cleanup3_crit_edge.us ], [ 0, %for.cond1.preheader.us.preheader ]
  %mul.us = mul nsw i32 %i.024.us, %n
  br label %for.body4.us

for.body4.us:
  %j.022.us = phi i32 [ 0, %for.cond1.preheader.us ], [ %inc6.us, %for.body4.us ]
  %add.us = add nsw i32 %j.022.us, %mul.us
  %arrayidx.us = getelementptr inbounds i32, ptr %B, i32 %add.us
  %0 = load i32, ptr %arrayidx.us, align 4
  %arrayidx5.us = getelementptr inbounds i32, ptr %A, i32 %add.us
  store i32 %0, ptr %arrayidx5.us, align 4
  %inc6.us = add nuw nsw i32 %j.022.us, 1
  %exitcond = icmp ne i32 %inc6.us, %n
  br i1 %exitcond, label %for.body4.us, label %for.cond1.for.cond.cleanup3_crit_edge.us

for.cond1.for.cond.cleanup3_crit_edge.us:
  %inc8.us = add nuw nsw i32 %i.024.us, 1
  %cmp.us = icmp slt i32 %inc8.us, %n
  br i1 %cmp.us, label %for.cond1.preheader.us, label %for.cond.cleanup.loopexit

for.cond.cleanup.loopexit:
  br label %for.cond.cleanup

for.cond.cleanup:
  ret void
}
