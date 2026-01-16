; RUN: opt -passes=loop-vectorize -force-vector-interleave=1 -vectorize-vector-loops \
; RUN:   -mtriple=aarch64 -mattr=+sve2p1,+v8.6a  -S < %s | FileCheck %s

; Test how NEON intrinsic are re-vectorised using HVLA.

define void @uabd(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @uabd(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 8 x i16>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.uabd.nxv8i16(<vscale x 8 x i1> splat (i1 true), <vscale x 8 x i16> [[WIDE_LOAD]], <vscale x 8 x i16> splat (i16 1))
; CHECK:    store <vscale x 8 x i16> [[WIDE_OP]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <8 x i16>, ptr %b, i64 %indvars.iv
  %0 = load <8 x i16>, ptr %arrayidx, align 16
  %result = call <8 x i16> @llvm.aarch64.neon.uabd.v8i16(<8 x i16> %0, <8 x i16> splat (i16 1))
  %arrayidx2 = getelementptr inbounds <8 x i16>, ptr %a, i64 %indvars.iv
  store <8 x i16> %result, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @sabd(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @sabd(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 8 x i16>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.sabd.nxv8i16(<vscale x 8 x i1> splat (i1 true), <vscale x 8 x i16> [[WIDE_LOAD]], <vscale x 8 x i16> splat (i16 1))
; CHECK:    store <vscale x 8 x i16> [[WIDE_OP]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <8 x i16>, ptr %b, i64 %indvars.iv
  %0 = load <8 x i16>, ptr %arrayidx, align 16
  %result = call <8 x i16> @llvm.aarch64.neon.sabd.v8i16(<8 x i16> %0, <8 x i16> splat (i16 1))
  %arrayidx2 = getelementptr inbounds <8 x i16>, ptr %a, i64 %indvars.iv
  store <8 x i16> %result, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @fabd(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @fabd(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 8 x half>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 8 x half> @llvm.aarch64.sve.fabd.nxv8f16(<vscale x 8 x i1> splat (i1 true), <vscale x 8 x half> [[WIDE_LOAD]], <vscale x 8 x half> splat (half 1.000000e+00))
; CHECK:    store <vscale x 8 x half> [[WIDE_OP]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <8 x half>, ptr %b, i64 %indvars.iv
  %0 = load <8 x half>, ptr %arrayidx, align 16
  %result = call <8 x half> @llvm.aarch64.neon.fabd.v8f16(<8 x half> %0, <8 x half> splat (half 1.000000e+00))
  %arrayidx2 = getelementptr inbounds <8 x half>, ptr %a, i64 %indvars.iv
  store <8 x half> %result, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @facge(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @facge(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 8 x half>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.facge.nxv8f16(<vscale x 8 x i1> splat (i1 true), <vscale x 8 x half> [[WIDE_LOAD]], <vscale x 8 x half> splat (half 1.000000e+00))
; CHECK:    [[TMP:%.*]] = zext <vscale x 8 x i1> [[WIDE_OP]] to <vscale x 8 x i16>
; CHECK:    store <vscale x 8 x i16> [[TMP]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <8 x half>, ptr %b, i64 %indvars.iv
  %0 = load <8 x half>, ptr %arrayidx, align 16
  %result = call <8 x i16> @llvm.aarch64.neon.facge.v8i16.v8f16(<8 x half> %0, <8 x half> splat (half 1.000000e+00))
  %arrayidx2 = getelementptr inbounds <8 x i16>, ptr %a, i64 %indvars.iv
  store <8 x i16> %result, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @facgt(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @facgt(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 8 x half>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.facgt.nxv8f16(<vscale x 8 x i1> splat (i1 true), <vscale x 8 x half> [[WIDE_LOAD]], <vscale x 8 x half> splat (half 1.000000e+00))
; CHECK:    [[TMP:%.*]] = zext <vscale x 8 x i1> [[WIDE_OP]] to <vscale x 8 x i16>
; CHECK:    store <vscale x 8 x i16> [[TMP]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <8 x half>, ptr %b, i64 %indvars.iv
  %0 = load <8 x half>, ptr %arrayidx, align 16
  %result = call <8 x i16> @llvm.aarch64.neon.facgt.v8i16.v8f16(<8 x half> %0, <8 x half> splat (half 1.000000e+00))
  %arrayidx2 = getelementptr inbounds <8 x i16>, ptr %a, i64 %indvars.iv
  store <8 x i16> %result, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @fcvtzs(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @fcvtzs(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 8 x half>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.fcvtzs.nxv8i16.nxv8f16(<vscale x 8 x i16> poison, <vscale x 8 x i1> splat (i1 true), <vscale x 8 x half> [[WIDE_LOAD]])
; CHECK:    store <vscale x 8 x i16> [[WIDE_OP]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <8 x half>, ptr %b, i64 %indvars.iv
  %0 = load <8 x half>, ptr %arrayidx, align 16
  %result = call <8 x i16> @llvm.aarch64.neon.fcvtzs.v8i16.v8f16(<8 x half> %0)
  %arrayidx2 = getelementptr inbounds <8 x i16>, ptr %a, i64 %indvars.iv
  store <8 x i16> %result, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @fcvtzu(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @fcvtzu(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 8 x half>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.fcvtzu.nxv8i16.nxv8f16(<vscale x 8 x i16> poison, <vscale x 8 x i1> splat (i1 true), <vscale x 8 x half> [[WIDE_LOAD]])
; CHECK:    store <vscale x 8 x i16> [[WIDE_OP]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <8 x half>, ptr %b, i64 %indvars.iv
  %0 = load <8 x half>, ptr %arrayidx, align 16
  %result = call <8 x i16> @llvm.aarch64.neon.fcvtzu.v8i16.v8f16(<8 x half> %0)
  %arrayidx2 = getelementptr inbounds <8 x i16>, ptr %a, i64 %indvars.iv
  store <8 x i16> %result, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @frintx(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @frintx(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x float>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.frintx.nxv4f32(<vscale x 4 x float> poison, <vscale x 4 x i1> splat (i1 true), <vscale x 4 x float> [[WIDE_LOAD]])
; CHECK:    store <vscale x 4 x float> [[WIDE_OP]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <4 x float>, ptr %b, i64 %indvars.iv
  %0 = load <4 x float>, ptr %arrayidx, align 16
  %result = call <4 x float> @llvm.aarch64.neon.frint32x.v4f32(<4 x float> %0)
  %arrayidx2 = getelementptr inbounds <4 x float>, ptr %a, i64 %indvars.iv
  store <4 x float> %result, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @frintz(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @frintz(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x float>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.frintz.nxv4f32(<vscale x 4 x float> poison, <vscale x 4 x i1> splat (i1 true), <vscale x 4 x float> [[WIDE_LOAD]])
; CHECK:    store <vscale x 4 x float> [[WIDE_OP]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <4 x float>, ptr %b, i64 %indvars.iv
  %0 = load <4 x float>, ptr %arrayidx, align 16
  %result = call <4 x float> @llvm.aarch64.neon.frint32z.v4f32(<4 x float> %0)
  %arrayidx2 = getelementptr inbounds <4 x float>, ptr %a, i64 %indvars.iv
  store <4 x float> %result, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @frecpe(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @frecpe(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x float>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.frecpe.x.nxv4f32(<vscale x 4 x float> [[WIDE_LOAD]])
; CHECK:    store <vscale x 4 x float> [[WIDE_OP]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <4 x float>, ptr %b, i64 %indvars.iv
  %0 = load <4 x float>, ptr %arrayidx, align 16
  %result = call <4 x float> @llvm.aarch64.neon.frecpe.v4f32(<4 x float> %0)
  %arrayidx2 = getelementptr inbounds <4 x float>, ptr %a, i64 %indvars.iv
  store <4 x float> %result, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @frecps(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @frecps(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x float>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.frecps.x.nxv4f32(<vscale x 4 x float> [[WIDE_LOAD]], <vscale x 4 x float> splat (float 1.000000e+00))
; CHECK:    store <vscale x 4 x float> [[WIDE_OP]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <4 x float>, ptr %b, i64 %indvars.iv
  %0 = load <4 x float>, ptr %arrayidx, align 16
  %result = call <4 x float> @llvm.aarch64.neon.frecps.v4f32(<4 x float> %0, <4 x float> splat (float 1.0))
  %arrayidx2 = getelementptr inbounds <4 x float>, ptr %a, i64 %indvars.iv
  store <4 x float> %result, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @frsqrte(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @frsqrte(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x float>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.frsqrte.x.nxv4f32(<vscale x 4 x float> [[WIDE_LOAD]])
; CHECK:    store <vscale x 4 x float> [[WIDE_OP]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <4 x float>, ptr %b, i64 %indvars.iv
  %0 = load <4 x float>, ptr %arrayidx, align 16
  %result = call <4 x float> @llvm.aarch64.neon.frsqrte.v4f32(<4 x float> %0)
  %arrayidx2 = getelementptr inbounds <4 x float>, ptr %a, i64 %indvars.iv
  store <4 x float> %result, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @frsqrts(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @frsqrts(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x float>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 4 x float> @llvm.aarch64.sve.frsqrts.x.nxv4f32(<vscale x 4 x float> [[WIDE_LOAD]], <vscale x 4 x float> splat (float 1.000000e+00))
; CHECK:    store <vscale x 4 x float> [[WIDE_OP]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <4 x float>, ptr %b, i64 %indvars.iv
  %0 = load <4 x float>, ptr %arrayidx, align 16
  %result = call <4 x float> @llvm.aarch64.neon.frsqrts.v4f32(<4 x float> %0, <4 x float> splat (float 1.0))
  %arrayidx2 = getelementptr inbounds <4 x float>, ptr %a, i64 %indvars.iv
  store <4 x float> %result, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @abs(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @abs(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 8 x i16>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.abs.nxv8i16(<vscale x 8 x i16> [[WIDE_LOAD]], <vscale x 8 x i1> splat (i1 true), <vscale x 8 x i16> [[WIDE_LOAD]])
; CHECK:    store <vscale x 8 x i16> [[WIDE_OP]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <8 x i16>, ptr %b, i64 %indvars.iv
  %0 = load <8 x i16>, ptr %arrayidx, align 16
  %result = call <8 x i16> @llvm.aarch64.neon.abs.v8i16(<8 x i16> %0)
  %arrayidx2 = getelementptr inbounds <8 x i16>, ptr %a, i64 %indvars.iv
  store <8 x i16> %result, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @sqabs(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @sqabs(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 8 x i16>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 8 x i16> @llvm.aarch64.sve.sqabs.nxv8i16(<vscale x 8 x i16> [[WIDE_LOAD]], <vscale x 8 x i1> splat (i1 true), <vscale x 8 x i16> [[WIDE_LOAD]])
; CHECK:    store <vscale x 8 x i16> [[WIDE_OP]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <8 x i16>, ptr %b, i64 %indvars.iv
  %0 = load <8 x i16>, ptr %arrayidx, align 16
  %result = call <8 x i16> @llvm.aarch64.neon.sqabs.v8i16(<8 x i16> %0)
  %arrayidx2 = getelementptr inbounds <8 x i16>, ptr %a, i64 %indvars.iv
  store <8 x i16> %result, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @uaddlp(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @uaddlp(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 8 x i16>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.uadalp.nxv4i32(<vscale x 4 x i1> splat (i1 true), <vscale x 4 x i32> zeroinitializer, <vscale x 8 x i16> [[WIDE_LOAD]])
; CHECK:    store <vscale x 4 x i32> [[WIDE_OP]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <8 x i16>, ptr %b, i64 %indvars.iv
  %0 = load <8 x i16>, ptr %arrayidx, align 16
  %vpaddl1.i = tail call <4 x i32> @llvm.aarch64.neon.uaddlp.v4i32.v8i16(<8 x i16> %0)
  %arrayidx2 = getelementptr inbounds <4 x i32>, ptr %a, i64 %indvars.iv
  store <4 x i32> %vpaddl1.i, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @saddlp(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @saddlp(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 8 x i16>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sadalp.nxv4i32(<vscale x 4 x i1> splat (i1 true), <vscale x 4 x i32> zeroinitializer, <vscale x 8 x i16> [[WIDE_LOAD]])
; CHECK:    store <vscale x 4 x i32> [[WIDE_OP]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <8 x i16>, ptr %b, i64 %indvars.iv
  %0 = load <8 x i16>, ptr %arrayidx, align 16
  %vpaddl1.i = tail call <4 x i32> @llvm.aarch64.neon.saddlp.v4i32.v8i16(<8 x i16> %0)
  %arrayidx2 = getelementptr inbounds <4 x i32>, ptr %a, i64 %indvars.iv
  store <4 x i32> %vpaddl1.i, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @udot(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @udot(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 16 x i8>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.udot.nxv4i32(<vscale x 4 x i32> zeroinitializer, <vscale x 16 x i8> [[WIDE_LOAD]], <vscale x 16 x i8> [[WIDE_LOAD]])
; CHECK:    store <vscale x 4 x i32> [[WIDE_OP]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <16 x i8>, ptr %b, i64 %indvars.iv
  %0 = load <16 x i8>, ptr %arrayidx, align 16
  %vpaddl1.i = tail call <4 x i32> @llvm.aarch64.neon.udot.v4i32.v16i8(<4 x i32> zeroinitializer, <16 x i8> %0, <16 x i8> %0)
  %arrayidx2 = getelementptr inbounds <4 x i32>, ptr %a, i64 %indvars.iv
  store <4 x i32> %vpaddl1.i, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @sdot(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @sdot(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 16 x i8>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sdot.nxv4i32(<vscale x 4 x i32> zeroinitializer, <vscale x 16 x i8> [[WIDE_LOAD]], <vscale x 16 x i8> [[WIDE_LOAD]])
; CHECK:    store <vscale x 4 x i32> [[WIDE_OP]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <16 x i8>, ptr %b, i64 %indvars.iv
  %0 = load <16 x i8>, ptr %arrayidx, align 16
  %vpaddl1.i = tail call <4 x i32> @llvm.aarch64.neon.sdot.v4i32.v16i8(<4 x i32> zeroinitializer, <16 x i8> %0, <16 x i8> %0)
  %arrayidx2 = getelementptr inbounds <4 x i32>, ptr %a, i64 %indvars.iv
  store <4 x i32> %vpaddl1.i, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @usdot(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @usdot(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 16 x i8>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.usdot.nxv4i32(<vscale x 4 x i32> zeroinitializer, <vscale x 16 x i8> [[WIDE_LOAD]], <vscale x 16 x i8> [[WIDE_LOAD]])
; CHECK:    store <vscale x 4 x i32> [[WIDE_OP]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <16 x i8>, ptr %b, i64 %indvars.iv
  %0 = load <16 x i8>, ptr %arrayidx, align 16
  %vpaddl1.i = tail call <4 x i32> @llvm.aarch64.neon.usdot.v4i32.v16i8(<4 x i32> zeroinitializer, <16 x i8> %0, <16 x i8> %0)
  %arrayidx2 = getelementptr inbounds <4 x i32>, ptr %a, i64 %indvars.iv
  store <4 x i32> %vpaddl1.i, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @uhadd(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @uhadd(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x i32>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.uhadd.nxv4i32(<vscale x 4 x i1> splat (i1 true), <vscale x 4 x i32> [[WIDE_LOAD]], <vscale x 4 x i32> splat (i32 1))
; CHECK:    store <vscale x 4 x i32> [[WIDE_OP]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <4 x i32>, ptr %b, i64 %indvars.iv
  %0 = load <4 x i32>, ptr %arrayidx, align 16
  %vpaddl1.i = tail call <4 x i32> @llvm.aarch64.neon.uhadd.v4i32(<4 x i32> %0, <4 x i32> splat (i32 1))
  %arrayidx2 = getelementptr inbounds <4 x i32>, ptr %a, i64 %indvars.iv
  store <4 x i32> %vpaddl1.i, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @shadd(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @shadd(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x i32>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.shadd.nxv4i32(<vscale x 4 x i1> splat (i1 true), <vscale x 4 x i32> [[WIDE_LOAD]], <vscale x 4 x i32> splat (i32 1))
; CHECK:    store <vscale x 4 x i32> [[WIDE_OP]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <4 x i32>, ptr %b, i64 %indvars.iv
  %0 = load <4 x i32>, ptr %arrayidx, align 16
  %vpaddl1.i = tail call <4 x i32> @llvm.aarch64.neon.shadd.v4i32(<4 x i32> %0, <4 x i32> splat (i32 1))
  %arrayidx2 = getelementptr inbounds <4 x i32>, ptr %a, i64 %indvars.iv
  store <4 x i32> %vpaddl1.i, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @uhsub(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @uhsub(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x i32>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.uhsub.nxv4i32(<vscale x 4 x i1> splat (i1 true), <vscale x 4 x i32> [[WIDE_LOAD]], <vscale x 4 x i32> splat (i32 1))
; CHECK:    store <vscale x 4 x i32> [[WIDE_OP]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <4 x i32>, ptr %b, i64 %indvars.iv
  %0 = load <4 x i32>, ptr %arrayidx, align 16
  %vpaddl1.i = tail call <4 x i32> @llvm.aarch64.neon.uhsub.v4i32(<4 x i32> %0, <4 x i32> splat (i32 1))
  %arrayidx2 = getelementptr inbounds <4 x i32>, ptr %a, i64 %indvars.iv
  store <4 x i32> %vpaddl1.i, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @shsub(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @shsub(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x i32>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.shsub.nxv4i32(<vscale x 4 x i1> splat (i1 true), <vscale x 4 x i32> [[WIDE_LOAD]], <vscale x 4 x i32> splat (i32 1))
; CHECK:    store <vscale x 4 x i32> [[WIDE_OP]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <4 x i32>, ptr %b, i64 %indvars.iv
  %0 = load <4 x i32>, ptr %arrayidx, align 16
  %vpaddl1.i = tail call <4 x i32> @llvm.aarch64.neon.shsub.v4i32(<4 x i32> %0, <4 x i32> splat (i32 1))
  %arrayidx2 = getelementptr inbounds <4 x i32>, ptr %a, i64 %indvars.iv
  store <4 x i32> %vpaddl1.i, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @uqadd(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @uqadd(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x i32>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.uqadd.nxv4i32(<vscale x 4 x i1> splat (i1 true), <vscale x 4 x i32> [[WIDE_LOAD]], <vscale x 4 x i32> splat (i32 1))
; CHECK:    store <vscale x 4 x i32> [[WIDE_OP]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <4 x i32>, ptr %b, i64 %indvars.iv
  %0 = load <4 x i32>, ptr %arrayidx, align 16
  %vpaddl1.i = tail call <4 x i32> @llvm.aarch64.neon.uqadd.v4i32(<4 x i32> %0, <4 x i32> splat (i32 1))
  %arrayidx2 = getelementptr inbounds <4 x i32>, ptr %a, i64 %indvars.iv
  store <4 x i32> %vpaddl1.i, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @sqadd(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @sqadd(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x i32>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sqadd.nxv4i32(<vscale x 4 x i1> splat (i1 true), <vscale x 4 x i32> [[WIDE_LOAD]], <vscale x 4 x i32> splat (i32 1))
; CHECK:    store <vscale x 4 x i32> [[WIDE_OP]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <4 x i32>, ptr %b, i64 %indvars.iv
  %0 = load <4 x i32>, ptr %arrayidx, align 16
  %vpaddl1.i = tail call <4 x i32> @llvm.aarch64.neon.sqadd.v4i32(<4 x i32> %0, <4 x i32> splat (i32 1))
  %arrayidx2 = getelementptr inbounds <4 x i32>, ptr %a, i64 %indvars.iv
  store <4 x i32> %vpaddl1.i, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @uqsub(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @uqsub(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x i32>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.uqsub.nxv4i32(<vscale x 4 x i1> splat (i1 true), <vscale x 4 x i32> [[WIDE_LOAD]], <vscale x 4 x i32> splat (i32 1))
; CHECK:    store <vscale x 4 x i32> [[WIDE_OP]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <4 x i32>, ptr %b, i64 %indvars.iv
  %0 = load <4 x i32>, ptr %arrayidx, align 16
  %vpaddl1.i = tail call <4 x i32> @llvm.aarch64.neon.uqsub.v4i32(<4 x i32> %0, <4 x i32> splat (i32 1))
  %arrayidx2 = getelementptr inbounds <4 x i32>, ptr %a, i64 %indvars.iv
  store <4 x i32> %vpaddl1.i, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @sqsub(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @sqsub(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x i32>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sqsub.nxv4i32(<vscale x 4 x i1> splat (i1 true), <vscale x 4 x i32> [[WIDE_LOAD]], <vscale x 4 x i32> splat (i32 1))
; CHECK:    store <vscale x 4 x i32> [[WIDE_OP]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <4 x i32>, ptr %b, i64 %indvars.iv
  %0 = load <4 x i32>, ptr %arrayidx, align 16
  %vpaddl1.i = tail call <4 x i32> @llvm.aarch64.neon.sqsub.v4i32(<4 x i32> %0, <4 x i32> splat (i32 1))
  %arrayidx2 = getelementptr inbounds <4 x i32>, ptr %a, i64 %indvars.iv
  store <4 x i32> %vpaddl1.i, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @sqdmulh(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @sqdmulh(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x i32>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sqdmulh.nxv4i32(<vscale x 4 x i32> [[WIDE_LOAD]], <vscale x 4 x i32> splat (i32 3))
; CHECK:    store <vscale x 4 x i32> [[WIDE_OP]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <4 x i32>, ptr %b, i64 %indvars.iv
  %0 = load <4 x i32>, ptr %arrayidx, align 16
  %vpaddl1.i = tail call <4 x i32> @llvm.aarch64.neon.sqdmulh.v4i32(<4 x i32> %0, <4 x i32> splat (i32 3))
  %arrayidx2 = getelementptr inbounds <4 x i32>, ptr %a, i64 %indvars.iv
  store <4 x i32> %vpaddl1.i, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @sqrdmulh(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @sqrdmulh(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x i32>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sqrdmulh.nxv4i32(<vscale x 4 x i32> [[WIDE_LOAD]], <vscale x 4 x i32> splat (i32 3))
; CHECK:    store <vscale x 4 x i32> [[WIDE_OP]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <4 x i32>, ptr %b, i64 %indvars.iv
  %0 = load <4 x i32>, ptr %arrayidx, align 16
  %vpaddl1.i = tail call <4 x i32> @llvm.aarch64.neon.sqrdmulh.v4i32(<4 x i32> %0, <4 x i32> splat (i32 3))
  %arrayidx2 = getelementptr inbounds <4 x i32>, ptr %a, i64 %indvars.iv
  store <4 x i32> %vpaddl1.i, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @sqshl(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @sqshl(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x i32>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sqshl.nxv4i32(<vscale x 4 x i1> splat (i1 true), <vscale x 4 x i32> [[WIDE_LOAD]], <vscale x 4 x i32> splat (i32 3))
; CHECK:    store <vscale x 4 x i32> [[WIDE_OP]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <4 x i32>, ptr %b, i64 %indvars.iv
  %0 = load <4 x i32>, ptr %arrayidx, align 16
  %vpaddl1.i = tail call <4 x i32> @llvm.aarch64.neon.sqshl.v4i32(<4 x i32> %0, <4 x i32> splat (i32 3))
  %arrayidx2 = getelementptr inbounds <4 x i32>, ptr %a, i64 %indvars.iv
  store <4 x i32> %vpaddl1.i, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @sqshlu(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @sqshlu(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x i32>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sqshlu.nxv4i32(<vscale x 4 x i1> splat (i1 true), <vscale x 4 x i32> [[WIDE_LOAD]], i32 1)
; CHECK:    store <vscale x 4 x i32> [[WIDE_OP]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <4 x i32>, ptr %b, i64 %indvars.iv
  %0 = load <4 x i32>, ptr %arrayidx, align 16
  %vpaddl1.i = tail call <4 x i32> @llvm.aarch64.neon.sqshlu.v4i32(<4 x i32> %0, <4 x i32> <i32 1, i32 1, i32 1, i32 1>)
  %arrayidx2 = getelementptr inbounds <4 x i32>, ptr %a, i64 %indvars.iv
  store <4 x i32> %vpaddl1.i, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @uqshl(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @uqshl(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x i32>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.uqshl.nxv4i32(<vscale x 4 x i1> splat (i1 true), <vscale x 4 x i32> [[WIDE_LOAD]], <vscale x 4 x i32> splat (i32 3))
; CHECK:    store <vscale x 4 x i32> [[WIDE_OP]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <4 x i32>, ptr %b, i64 %indvars.iv
  %0 = load <4 x i32>, ptr %arrayidx, align 16
  %vpaddl1.i = tail call <4 x i32> @llvm.aarch64.neon.uqshl.v4i32(<4 x i32> %0, <4 x i32> splat (i32 3))
  %arrayidx2 = getelementptr inbounds <4 x i32>, ptr %a, i64 %indvars.iv
  store <4 x i32> %vpaddl1.i, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @srshl(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @srshl(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x i32>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.srshl.nxv4i32(<vscale x 4 x i1> splat (i1 true), <vscale x 4 x i32> [[WIDE_LOAD]], <vscale x 4 x i32> splat (i32 3))
; CHECK:    store <vscale x 4 x i32> [[WIDE_OP]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <4 x i32>, ptr %b, i64 %indvars.iv
  %0 = load <4 x i32>, ptr %arrayidx, align 16
  %vpaddl1.i = tail call <4 x i32> @llvm.aarch64.neon.srshl.v4i32(<4 x i32> %0, <4 x i32> splat (i32 3))
  %arrayidx2 = getelementptr inbounds <4 x i32>, ptr %a, i64 %indvars.iv
  store <4 x i32> %vpaddl1.i, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @sqrshl(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @sqrshl(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x i32>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sqrshl.nxv4i32(<vscale x 4 x i1> splat (i1 true), <vscale x 4 x i32> [[WIDE_LOAD]], <vscale x 4 x i32> splat (i32 3))
; CHECK:    store <vscale x 4 x i32> [[WIDE_OP]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <4 x i32>, ptr %b, i64 %indvars.iv
  %0 = load <4 x i32>, ptr %arrayidx, align 16
  %vpaddl1.i = tail call <4 x i32> @llvm.aarch64.neon.sqrshl.v4i32(<4 x i32> %0, <4 x i32> splat (i32 3))
  %arrayidx2 = getelementptr inbounds <4 x i32>, ptr %a, i64 %indvars.iv
  store <4 x i32> %vpaddl1.i, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @sshl(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @sshl(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x i32>
; CHECK:    [[SHL_MASK:%.*]] = icmp sle <vscale x 4 x i32> [[WIDE_LOAD]], zeroinitializer
; CHECK:    [[TMP3:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.srshl.nxv4i32(<vscale x 4 x i1> [[SHL_MASK]], <vscale x 4 x i32> splat (i32 8), <vscale x 4 x i32> [[WIDE_LOAD]])
; CHECK:    [[SHR_MASK:%.*]] = call <vscale x 4 x i1> @llvm.ctlz.nxv4i1(<vscale x 4 x i1> [[SHL_MASK]], i1 true)
; CHECK:    [[WIDE_BIDIR_SHL:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sqshl.nxv4i32(<vscale x 4 x i1> [[SHR_MASK]], <vscale x 4 x i32> [[TMP3]], <vscale x 4 x i32> [[WIDE_LOAD]])
; CHECK:    store <vscale x 4 x i32> [[WIDE_BIDIR_SHL]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <4 x i32>, ptr %b, i64 %indvars.iv
  %0 = load <4 x i32>, ptr %arrayidx, align 16
  %vpaddl1.i = tail call <4 x i32> @llvm.aarch64.neon.sshl.v4i32(<4 x i32> splat (i32 8), <4 x i32> %0)
  %arrayidx2 = getelementptr inbounds <4 x i32>, ptr %a, i64 %indvars.iv
  store <4 x i32> %vpaddl1.i, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @ushl(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @ushl(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x i32>
; CHECK:    [[SHL_MASK:%.*]] = icmp sle <vscale x 4 x i32> [[WIDE_LOAD]], zeroinitializer
; CHECK:    [[TMP3:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.urshl.nxv4i32(<vscale x 4 x i1> [[SHL_MASK]], <vscale x 4 x i32> splat (i32 8), <vscale x 4 x i32> [[WIDE_LOAD]])
; CHECK:    [[SHR_MASK:%.*]] = call <vscale x 4 x i1> @llvm.ctlz.nxv4i1(<vscale x 4 x i1> [[SHL_MASK]], i1 true)
; CHECK:    [[WIDE_BIDIR_SHL:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.uqshl.nxv4i32(<vscale x 4 x i1> [[SHR_MASK]], <vscale x 4 x i32> [[TMP3]], <vscale x 4 x i32> [[WIDE_LOAD]])
; CHECK:    store <vscale x 4 x i32> [[WIDE_BIDIR_SHL]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <4 x i32>, ptr %b, i64 %indvars.iv
  %0 = load <4 x i32>, ptr %arrayidx, align 16
  %vpaddl1.i = tail call <4 x i32> @llvm.aarch64.neon.ushl.v4i32(<4 x i32> splat (i32 8), <4 x i32> %0)
  %arrayidx2 = getelementptr inbounds <4 x i32>, ptr %a, i64 %indvars.iv
  store <4 x i32> %vpaddl1.i, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @vsli(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @vsli(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x i32>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sli.nxv4i32(<vscale x 4 x i32> splat (i32 1), <vscale x 4 x i32> [[WIDE_LOAD]], i32 2)
; CHECK:    store <vscale x 4 x i32> [[WIDE_OP]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <4 x i32>, ptr %b, i64 %indvars.iv
  %0 = load <4 x i32>, ptr %arrayidx, align 16
  %vpaddl1.i = tail call <4 x i32> @llvm.aarch64.neon.vsli.v4i32(<4 x i32> splat (i32 1), <4 x i32> %0, i32 2)
  %arrayidx2 = getelementptr inbounds <4 x i32>, ptr %a, i64 %indvars.iv
  store <4 x i32> %vpaddl1.i, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @vsri(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @vsri(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x i32>
; CHECK:    [[WIDE_OP:%.*]] = call <vscale x 4 x i32> @llvm.aarch64.sve.sri.nxv4i32(<vscale x 4 x i32> splat (i32 1), <vscale x 4 x i32> [[WIDE_LOAD]], i32 2)
; CHECK:    store <vscale x 4 x i32> [[WIDE_OP]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <4 x i32>, ptr %b, i64 %indvars.iv
  %0 = load <4 x i32>, ptr %arrayidx, align 16
  %vpaddl1.i = tail call <4 x i32> @llvm.aarch64.neon.vsri.v4i32(<4 x i32> splat (i32 1), <4 x i32> %0, i32 2)
  %arrayidx2 = getelementptr inbounds <4 x i32>, ptr %a, i64 %indvars.iv
  store <4 x i32> %vpaddl1.i, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @addp(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @addp(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 4 x i32>
; CHECK:    [[TMP3:%.*]] = call <vscale x 4 x i32> @llvm.vector.segmented.shuffle.nxv4i32.v4i32(<vscale x 4 x i32> splat (i32 8), <vscale x 4 x i32> [[WIDE_LOAD]], <4 x i32> <i32 0, i32 2, i32 4, i32 6>)
; CHECK:    [[TMP4:%.*]] = call <vscale x 4 x i32> @llvm.vector.segmented.shuffle.nxv4i32.v4i32(<vscale x 4 x i32> splat (i32 8), <vscale x 4 x i32> [[WIDE_LOAD]], <4 x i32> <i32 1, i32 3, i32 5, i32 7>)
; CHECK:    [[TMP5:%.*]] = add <vscale x 4 x i32> [[TMP3]], [[TMP4]]
; CHECK:    store <vscale x 4 x i32> [[TMP5]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <4 x i32>, ptr %b, i64 %indvars.iv
  %0 = load <4 x i32>, ptr %arrayidx, align 16
  %vpaddl1.i = tail call <4 x i32> @llvm.aarch64.neon.addp.v4i32(<4 x i32> splat (i32 8), <4 x i32> %0)
  %arrayidx2 = getelementptr inbounds <4 x i32>, ptr %a, i64 %indvars.iv
  store <4 x i32> %vpaddl1.i, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @addp_64(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @addp_64(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 2 x i64>
; CHECK:    [[RES:%.*]] = call <vscale x 2 x i64> @llvm.aarch64.sve.addp.nxv2i64(<vscale x 2 x i1> splat (i1 true), <vscale x 2 x i64> splat (i64 8), <vscale x 2 x i64> [[WIDE_LOAD]])
; CHECK:    store <vscale x 2 x i64> [[RES]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <2 x i64>, ptr %b, i64 %indvars.iv
  %0 = load <2 x i64>, ptr %arrayidx, align 16
  %vpaddl1.i = tail call <2 x i64> @llvm.aarch64.neon.addp.v2i64(<2 x i64> splat (i64 8), <2 x i64> %0)
  %arrayidx2 = getelementptr inbounds <2 x i64>, ptr %a, i64 %indvars.iv
  store <2 x i64> %vpaddl1.i, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}


define void @tbl1(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @tbl1(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 16 x i8>
; CHECK:    [[RES:%.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.tblq.nxv16i8(<vscale x 16 x i8> [[WIDE_LOAD]], <vscale x 16 x i8> [[WIDE_LOAD]])
; CHECK:    store <vscale x 16 x i8> [[RES]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <16 x i8>, ptr %b, i64 %indvars.iv
  %0 = load <16 x i8>, ptr %arrayidx, align 16
  %res = tail call <16 x i8> @llvm.aarch64.neon.tbl1.v16i8(<16 x i8> %0, <16 x i8> %0)
  %arrayidx2 = getelementptr inbounds <16 x i8>, ptr %a, i64 %indvars.iv
  store <16 x i8> %res, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @tbl2(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @tbl2(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 16 x i8>
; CHECK:    [[RES:%.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.tblq.nxv16i8(<vscale x 16 x i8> [[WIDE_LOAD]], <vscale x 16 x i8> [[WIDE_LOAD]])
; CHECK:    [[TBX_MASK:%.*]] = sub <vscale x 16 x i8> [[WIDE_LOAD]], splat (i8 16)
; CHECK:    [[RES2:%.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.tbxq.nxv16i8(<vscale x 16 x i8> [[RES]], <vscale x 16 x i8> splat (i8 4), <vscale x 16 x i8> [[TBX_MASK]])
; CHECK:    store <vscale x 16 x i8> [[RES2]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <16 x i8>, ptr %b, i64 %indvars.iv
  %0 = load <16 x i8>, ptr %arrayidx, align 16
  %res = tail call <16 x i8> @llvm.aarch64.neon.tbl2.v16i8(<16 x i8> %0, <16 x i8> splat (i8 4), <16 x i8> %0)
  %arrayidx2 = getelementptr inbounds <16 x i8>, ptr %a, i64 %indvars.iv
  store <16 x i8> %res, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

define void @tbl3(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
; CHECK-LABEL: define void @tbl3(
; CHECK:    [[WIDE_LOAD:%.*]] = load <vscale x 16 x i8>
; CHECK:    [[RES:%.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.tblq.nxv16i8(<vscale x 16 x i8> [[WIDE_LOAD]], <vscale x 16 x i8> [[WIDE_LOAD]])
; CHECK:    [[TBX_MASK:%.*]] = sub <vscale x 16 x i8> [[WIDE_LOAD]], splat (i8 16)
; CHECK:    [[RES2:%.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.tbxq.nxv16i8(<vscale x 16 x i8> [[RES]], <vscale x 16 x i8> splat (i8 4), <vscale x 16 x i8> [[TBX_MASK]])
; CHECK:    [[TBX_MASK2:%.*]] = sub <vscale x 16 x i8> [[TBX_MASK]], splat (i8 16)
; CHECK:    [[RES3:%.*]] = call <vscale x 16 x i8> @llvm.aarch64.sve.tbxq.nxv16i8(<vscale x 16 x i8> [[RES2]], <vscale x 16 x i8> splat (i8 6), <vscale x 16 x i8> [[TBX_MASK2]])
; CHECK:    store <vscale x 16 x i8> [[RES3]]
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <16 x i8>, ptr %b, i64 %indvars.iv
  %0 = load <16 x i8>, ptr %arrayidx, align 16
  %res = tail call <16 x i8> @llvm.aarch64.neon.tbl3.v16i8(<16 x i8> %0, <16 x i8> splat (i8 4), <16 x i8> splat (i8 6), <16 x i8> %0)
  %arrayidx2 = getelementptr inbounds <16 x i8>, ptr %a, i64 %indvars.iv
  store <16 x i8> %res, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
