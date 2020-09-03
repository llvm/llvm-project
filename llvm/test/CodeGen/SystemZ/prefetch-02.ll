; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 -prefetch-distance=100 \
; RUN:   -stop-after=loop-data-prefetch | FileCheck %s -check-prefix=FAR-PREFETCH
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 -prefetch-distance=20 \
; RUN:   -stop-after=loop-data-prefetch | FileCheck %s -check-prefix=NEAR-PREFETCH
;
; Check that prefetches are not emitted when the known constant trip count of
; the loop is smaller than the estimated "iterations ahead" of the prefetch.
;
; FAR-PREFETCH-LABEL: fun
; FAR-PREFETCH-NOT: call void @llvm.prefetch

; NEAR-PREFETCH-LABEL: fun
; NEAR-PREFETCH: call void @llvm.prefetch


define void @fun(i32* nocapture %Src, i32* nocapture readonly %Dst) {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next.9, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %Dst, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %Src, i64 %indvars.iv
  store i32 %0, i32* %arrayidx2, align 4
  %indvars.iv.next.9 = add nuw nsw i64 %indvars.iv, 1600
  %cmp.9 = icmp ult i64 %indvars.iv.next.9, 11200
  br i1 %cmp.9, label %for.body, label %for.cond.cleanup
}

