; RUN: opt -passes=loop-vectorize -force-vector-interleave=1 -vectorize-vector-loops \
; RUN:   -mtriple=aarch64 -mattr=+sve2p1 -S < %s | FileCheck %s

; Test how NEON intrinsic are re-vectorised using HVLA.


define void @rshrn(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @rshrn(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x i32>
; CHECK:    [[TMP3:%.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.rshrnb.nxv4i32(<vscale x 4 x i32> [[WIDE_LOAD]], i32 1)
; CHECK:    [[TMP4:%.*]] = call { <vscale x 4 x i16>, <vscale x 4 x i16> } @llvm.vector.deinterleave2.nxv8i16(<vscale x 8 x i16> [[TMP3]])
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
  %vpaddl1.i = tail call <4 x i16> @llvm.aarch64.neon.rshrn.v4i16(<4 x i32> %0, i32 1)
  %arrayidx2 = getelementptr inbounds <4 x i16>, ptr %a, i64 %indvars.iv
  store <4 x i16> %vpaddl1.i, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @sqrshrn(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @sqrshrn(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x i32>
; CHECK:    [[TMP3:%.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.sqrshrnb.nxv4i32(<vscale x 4 x i32> [[WIDE_LOAD]], i32 1)
; CHECK:    [[TMP4:%.*]] = call { <vscale x 4 x i16>, <vscale x 4 x i16> } @llvm.vector.deinterleave2.nxv8i16(<vscale x 8 x i16> [[TMP3]])
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
  %vpaddl1.i = tail call <4 x i16> @llvm.aarch64.neon.sqrshrn.v4i16(<4 x i32> %0, i32 1)
  %arrayidx2 = getelementptr inbounds <4 x i16>, ptr %a, i64 %indvars.iv
  store <4 x i16> %vpaddl1.i, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @sqrshrun(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @sqrshrun(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x i32>
; CHECK:    [[TMP3:%.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.sqrshrunb.nxv4i32(<vscale x 4 x i32> [[WIDE_LOAD]], i32 1)
; CHECK:    [[TMP4:%.*]] = call { <vscale x 4 x i16>, <vscale x 4 x i16> } @llvm.vector.deinterleave2.nxv8i16(<vscale x 8 x i16> [[TMP3]])
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
  %vpaddl1.i = tail call <4 x i16> @llvm.aarch64.neon.sqrshrun.v4i16(<4 x i32> %0, i32 1)
  %arrayidx2 = getelementptr inbounds <4 x i16>, ptr %a, i64 %indvars.iv
  store <4 x i16> %vpaddl1.i, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @sqshrn(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @sqshrn(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x i32>
; CHECK:    [[TMP3:%.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.sqshrnb.nxv4i32(<vscale x 4 x i32> [[WIDE_LOAD]], i32 1)
; CHECK:    [[TMP4:%.*]] = call { <vscale x 4 x i16>, <vscale x 4 x i16> } @llvm.vector.deinterleave2.nxv8i16(<vscale x 8 x i16> [[TMP3]])
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
  %vpaddl1.i = tail call <4 x i16> @llvm.aarch64.neon.sqshrn.v4i16(<4 x i32> %0, i32 1)
  %arrayidx2 = getelementptr inbounds <4 x i16>, ptr %a, i64 %indvars.iv
  store <4 x i16> %vpaddl1.i, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @sqshrun(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @sqshrun(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x i32>
; CHECK:    [[TMP3:%.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.sqshrunb.nxv4i32(<vscale x 4 x i32> [[WIDE_LOAD]], i32 1)
; CHECK:    [[TMP4:%.*]] = call { <vscale x 4 x i16>, <vscale x 4 x i16> } @llvm.vector.deinterleave2.nxv8i16(<vscale x 8 x i16> [[TMP3]])
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
  %vpaddl1.i = tail call <4 x i16> @llvm.aarch64.neon.sqshrun.v4i16(<4 x i32> %0, i32 1)
  %arrayidx2 = getelementptr inbounds <4 x i16>, ptr %a, i64 %indvars.iv
  store <4 x i16> %vpaddl1.i, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @uqrshrn(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @uqrshrn(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x i32>
; CHECK:    [[TMP3:%.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.uqrshrnb.nxv4i32(<vscale x 4 x i32> [[WIDE_LOAD]], i32 1)
; CHECK:    [[TMP4:%.*]] = call { <vscale x 4 x i16>, <vscale x 4 x i16> } @llvm.vector.deinterleave2.nxv8i16(<vscale x 8 x i16> [[TMP3]])
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
  %vpaddl1.i = tail call <4 x i16> @llvm.aarch64.neon.uqrshrn.v4i16(<4 x i32> %0, i32 1)
  %arrayidx2 = getelementptr inbounds <4 x i16>, ptr %a, i64 %indvars.iv
  store <4 x i16> %vpaddl1.i, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @uqshrn(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @uqshrn(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x i32>
; CHECK:    [[TMP3:%.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.uqshrnb.nxv4i32(<vscale x 4 x i32> [[WIDE_LOAD]], i32 1)
; CHECK:    [[TMP4:%.*]] = call { <vscale x 4 x i16>, <vscale x 4 x i16> } @llvm.vector.deinterleave2.nxv8i16(<vscale x 8 x i16> [[TMP3]])
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
  %vpaddl1.i = tail call <4 x i16> @llvm.aarch64.neon.uqshrn.v4i16(<4 x i32> %0, i32 1)
  %arrayidx2 = getelementptr inbounds <4 x i16>, ptr %a, i64 %indvars.iv
  store <4 x i16> %vpaddl1.i, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
