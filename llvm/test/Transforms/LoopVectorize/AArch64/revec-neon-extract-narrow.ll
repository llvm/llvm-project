; RUN: opt -passes=loop-vectorize -force-vector-interleave=1 -vectorize-vector-loops \
; RUN:   -mtriple=aarch64 -mattr=+sve2p1 -S < %s | FileCheck %s

; Test how NEON intrinsic are re-vectorised using HVLA.


define void @sqxtn(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @sqxtn(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x i32>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.sqxtnb.nxv4i32(<vscale x 4 x i32> [[WIDE_LOAD]])
; CHECK:    [[TMP4:%.*]] = call { <vscale x 4 x i16>, <vscale x 4 x i16> } @llvm.vector.deinterleave2.nxv8i16(<vscale x 8 x i16> [[WIDE_OP]])
; CHECK:    [[TMP5:%.*]] = extractvalue { <vscale x 4 x i16>, <vscale x 4 x i16> } [[TMP4]], 0
; CHECK:    store <vscale x 4 x i16> [[TMP5]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <4 x i32>, ptr %b, i64 %indvars.iv
  %0 = load <4 x i32>, ptr %arrayidx, align 16
  %vpaddl1.i = tail call <4 x i16> @llvm.aarch64.neon.sqxtn.v4i16(<4 x i32> %0)
  %arrayidx2 = getelementptr inbounds <4 x i16>, ptr %a, i64 %indvars.iv
  store <4 x i16> %vpaddl1.i, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @sqxtun(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @sqxtun(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x i32>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.sqxtunb.nxv4i32(<vscale x 4 x i32> [[WIDE_LOAD]])
; CHECK:    [[TMP4:%.*]] = call { <vscale x 4 x i16>, <vscale x 4 x i16> } @llvm.vector.deinterleave2.nxv8i16(<vscale x 8 x i16> [[WIDE_OP]])
; CHECK:    [[TMP5:%.*]] = extractvalue { <vscale x 4 x i16>, <vscale x 4 x i16> } [[TMP4]], 0
; CHECK:    store <vscale x 4 x i16> [[TMP5]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <4 x i32>, ptr %b, i64 %indvars.iv
  %0 = load <4 x i32>, ptr %arrayidx, align 16
  %vpaddl1.i = tail call <4 x i16> @llvm.aarch64.neon.sqxtun.v4i16(<4 x i32> %0)
  %arrayidx2 = getelementptr inbounds <4 x i16>, ptr %a, i64 %indvars.iv
  store <4 x i16> %vpaddl1.i, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @uqxtn(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @uqxtn(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x i32>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.uqxtnb.nxv4i32(<vscale x 4 x i32> [[WIDE_LOAD]])
; CHECK:    [[TMP4:%.*]] = call { <vscale x 4 x i16>, <vscale x 4 x i16> } @llvm.vector.deinterleave2.nxv8i16(<vscale x 8 x i16> [[WIDE_OP]])
; CHECK:    [[TMP5:%.*]] = extractvalue { <vscale x 4 x i16>, <vscale x 4 x i16> } [[TMP4]], 0
; CHECK:    store <vscale x 4 x i16> [[TMP5]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <4 x i32>, ptr %b, i64 %indvars.iv
  %0 = load <4 x i32>, ptr %arrayidx, align 16
  %vpaddl1.i = tail call <4 x i16> @llvm.aarch64.neon.uqxtn.v4i16(<4 x i32> %0)
  %arrayidx2 = getelementptr inbounds <4 x i16>, ptr %a, i64 %indvars.iv
  store <4 x i16> %vpaddl1.i, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @fcvtxn(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @fcvtxn(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 2 x double>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.fcvtx.f32f64(<vscale x 4 x float> poison, <vscale x 2 x i1> splat (i1 true), <vscale x 2 x double> [[WIDE_LOAD]])
; CHECK:    [[TMP4:%.*]] = call { <vscale x 2 x float>, <vscale x 2 x float> } @llvm.vector.deinterleave2.nxv4f32(<vscale x 4 x float> [[WIDE_OP]])
; CHECK:    [[TMP5:%.*]] = extractvalue { <vscale x 2 x float>, <vscale x 2 x float> } [[TMP4]], 0
; CHECK:    store <vscale x 2 x float> [[TMP5]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <2 x double>, ptr %b, i64 %indvars.iv
  %0 = load <2 x double>, ptr %arrayidx, align 16
  %vpaddl1.i = tail call <2 x float> @llvm.aarch64.neon.fcvtxn.v2f64(<2 x double> %0)
  %arrayidx2 = getelementptr inbounds <2 x float>, ptr %a, i64 %indvars.iv
  store <2 x float> %vpaddl1.i, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
