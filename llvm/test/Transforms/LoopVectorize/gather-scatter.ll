; RUN: opt -passes=loop-vectorize -force-vector-width=4 -force-target-supports-gather-scatter-ops=true \
; RUN:     -S < %s | FileCheck %s --check-prefixes=CHECK,GATHER-SCATTER-ENABLED
; RUN: opt -passes=loop-vectorize -force-vector-width=4 -force-target-supports-gather-scatter-ops=false \
; RUN:     -S < %s | FileCheck %s --check-prefixes=CHECK,GATHER-SCATTER-DISABLED

; Generic test using for the force-target-supports-gather-scatter-ops option.

define void @gather_i32(ptr noalias nocapture readonly %a, ptr noalias nocapture readonly %b, ptr noalias nocapture %c, i64 %n) {
; CHECK-LABEL: define void @gather_i32(
; GATHER-SCATTER-ENABLED:         call <4 x float> @llvm.masked.gather.v4f32
; GATHER-SCATTER-DISABLED-NOT:    @llvm.masked.gather
entry:
  br label %for.body
for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i64, ptr %b, i64 %indvars.iv
  %0 = load i64, ptr %arrayidx, align 8
  %arrayidx3 = getelementptr inbounds float, ptr %a, i64 %0
  %1 = load float, ptr %arrayidx3, align 4
  %arrayidx5 = getelementptr inbounds float, ptr %c, i64 %indvars.iv
  store float %1, ptr %arrayidx5, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
for.cond.cleanup:
  ret void
}

define void @scatter_i32(ptr noalias nocapture readonly %a, ptr noalias nocapture readonly %b, ptr noalias nocapture %c, i64 %n) {
; CHECK-LABEL: define void @scatter_i32(
; GATHER-SCATTER-ENABLED:         call void @llvm.masked.scatter.v4f32
; GATHER-SCATTER-DISABLED-NOT:    @llvm.masked.scatter
entry:
  br label %for.body
for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i64, ptr %b, i64 %indvars.iv
  %0 = load i64, ptr %arrayidx, align 8
  %arrayidx3 = getelementptr inbounds float, ptr %a, i64 %indvars.iv
  %1 = load float, ptr %arrayidx3, align 4
  %arrayidx5 = getelementptr inbounds float, ptr %c, i64 %0
  store float %1, ptr %arrayidx5, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
for.cond.cleanup:
  ret void
}
