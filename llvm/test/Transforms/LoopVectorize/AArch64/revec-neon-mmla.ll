; RUN: opt -passes=loop-vectorize -force-vector-interleave=1 -vectorize-vector-loops \
; RUN:   -mtriple=aarch64 -mattr=+sve2p1,+i8mm -S < %s | FileCheck %s

; Test how NEON intrinsic are re-vectorised using HVLA.


define void @smmla(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @smmla(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 16 x i8>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.smmla.nxv4i32(<vscale x 4 x i32> splat (i32 8), <vscale x 16 x i8> [[WIDE_LOAD]], <vscale x 16 x i8> [[WIDE_LOAD]])
; CHECK:    store <vscale x 4 x i32> [[WIDE_OP]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <16 x i8>, ptr %b, i64 %indvars.iv
  %0 = load <16 x i8>, ptr %arrayidx, align 16
  %vpaddl1.i = tail call <4 x i32> @llvm.aarch64.neon.smmla.v4i32.v16i8(<4 x i32> splat (i32 8), <16 x i8> %0, <16 x i8> %0)
  %arrayidx2 = getelementptr inbounds <4 x i32>, ptr %a, i64 %indvars.iv
  store <4 x i32> %vpaddl1.i, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @ummla(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @ummla(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 16 x i8>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.ummla.nxv4i32(<vscale x 4 x i32> splat (i32 8), <vscale x 16 x i8> [[WIDE_LOAD]], <vscale x 16 x i8> [[WIDE_LOAD]])
; CHECK:    store <vscale x 4 x i32> [[WIDE_OP]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <16 x i8>, ptr %b, i64 %indvars.iv
  %0 = load <16 x i8>, ptr %arrayidx, align 16
  %vpaddl1.i = tail call <4 x i32> @llvm.aarch64.neon.ummla.v4i32.v16i8(<4 x i32> splat (i32 8), <16 x i8> %0, <16 x i8> %0)
  %arrayidx2 = getelementptr inbounds <4 x i32>, ptr %a, i64 %indvars.iv
  store <4 x i32> %vpaddl1.i, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @usmmla(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @usmmla(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 16 x i8>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.usmmla.nxv4i32(<vscale x 4 x i32> splat (i32 8), <vscale x 16 x i8> [[WIDE_LOAD]], <vscale x 16 x i8> [[WIDE_LOAD]])
; CHECK:    store <vscale x 4 x i32> [[WIDE_OP]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <16 x i8>, ptr %b, i64 %indvars.iv
  %0 = load <16 x i8>, ptr %arrayidx, align 16
  %vpaddl1.i = tail call <4 x i32> @llvm.aarch64.neon.usmmla.v4i32.v16i8(<4 x i32> splat (i32 8), <16 x i8> %0, <16 x i8> %0)
  %arrayidx2 = getelementptr inbounds <4 x i32>, ptr %a, i64 %indvars.iv
  store <4 x i32> %vpaddl1.i, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
