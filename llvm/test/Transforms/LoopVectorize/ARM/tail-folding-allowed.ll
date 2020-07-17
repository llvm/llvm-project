; RUN: opt -mtriple=thumbv8.1m.main-arm-eabihf -mattr=+mve.fp -loop-vectorize -tail-predication=enabled -S < %s | \
; RUN:  FileCheck %s

; Test that ARMTTIImpl::preferPredicateOverEpilogue triggers tail-folding.

define dso_local void @f1(i32* noalias nocapture %A, i32* noalias nocapture readonly %B, i32* noalias nocapture readonly %C, i32 %N) {
; CHECK-LABEL: f1(
; CHECK:       entry:
; CHECK:       @llvm.get.active.lane.mask
; CHECK:       }
entry:
  %cmp8 = icmp sgt i32 %N, 0
  br i1 %cmp8, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %i.09 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %B, i32 %i.09
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %C, i32 %i.09
  %1 = load i32, i32* %arrayidx1, align 4
  %add = add nsw i32 %1, %0
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i32 %i.09
  store i32 %add, i32* %arrayidx2, align 4
  %inc = add nuw nsw i32 %i.09, 1
  %exitcond.not = icmp eq i32 %inc, %N
  br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body
}
