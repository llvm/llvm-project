; RUN: opt -passes=loop-vectorize -force-vector-interleave=1 -vectorize-vector-loops -mtriple=aarch64 -mattr=+sve2p1 -S < %s | FileCheck %s

; Test that NEON rounding halving add intrinsics (urhadd/srhadd) can be
; successfully re-vectorized to SVE.

define void @urhadd_v16i8(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b, ptr nocapture noundef readonly %c) {
; CHECK-LABEL: define void @urhadd_v16i8(
; CHECK:    [[WIDE_LOAD_B:%.*]] = load <vscale x 16 x i8>
; CHECK:    [[WIDE_LOAD_C:%.*]] = load <vscale x 16 x i8>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.urhadd.nxv16i8(<vscale x 16 x i1> splat (i1 true), <vscale x 16 x i8> [[WIDE_LOAD_B]], <vscale x 16 x i8> [[WIDE_LOAD_C]])
; CHECK:    store <vscale x 16 x i8> [[WIDE_OP]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <16 x i8>, ptr %b, i64 %indvars.iv
  %0 = load <16 x i8>, ptr %arrayidx, align 16
  %arrayidx2 = getelementptr inbounds <16 x i8>, ptr %c, i64 %indvars.iv
  %1 = load <16 x i8>, ptr %arrayidx2, align 16
  %result = call <16 x i8> @llvm.aarch64.neon.urhadd.v16i8(<16 x i8> %0, <16 x i8> %1)
  %arrayidx3 = getelementptr inbounds <16 x i8>, ptr %a, i64 %indvars.iv
  store <16 x i8> %result, ptr %arrayidx3, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare <16 x i8> @llvm.aarch64.neon.urhadd.v16i8(<16 x i8>, <16 x i8>)

define void @srhadd_v16i8(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b, ptr nocapture noundef readonly %c) {
; CHECK-LABEL: define void @srhadd_v16i8(
; CHECK:    [[WIDE_LOAD_B:%.*]] = load <vscale x 16 x i8>
; CHECK:    [[WIDE_LOAD_C:%.*]] = load <vscale x 16 x i8>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.srhadd.nxv16i8(<vscale x 16 x i1> splat (i1 true), <vscale x 16 x i8> [[WIDE_LOAD_B]], <vscale x 16 x i8> [[WIDE_LOAD_C]])
; CHECK:    store <vscale x 16 x i8> [[WIDE_OP]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <16 x i8>, ptr %b, i64 %indvars.iv
  %0 = load <16 x i8>, ptr %arrayidx, align 16
  %arrayidx2 = getelementptr inbounds <16 x i8>, ptr %c, i64 %indvars.iv
  %1 = load <16 x i8>, ptr %arrayidx2, align 16
  %result = call <16 x i8> @llvm.aarch64.neon.srhadd.v16i8(<16 x i8> %0, <16 x i8> %1)
  %arrayidx3 = getelementptr inbounds <16 x i8>, ptr %a, i64 %indvars.iv
  store <16 x i8> %result, ptr %arrayidx3, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

declare <16 x i8> @llvm.aarch64.neon.srhadd.v16i8(<16 x i8>, <16 x i8>)
