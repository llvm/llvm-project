; RUN: opt -loop-vectorize -dce -instcombine -mtriple aarch64-linux-gnu -mattr=+sve -S %s -o - | FileCheck %s

define void @gather_nxv4i32_ind64(float* noalias nocapture readonly %a, i64* noalias nocapture readonly %b, float* noalias nocapture %c, i64 %n) {
; CHECK-LABEL: @gather_nxv4i32_ind64
; CHECK: vector.body:
; CHECK:   %[[IND:.*]] = load <vscale x 4 x i64>, <vscale x 4 x i64>*
; CHECK:   %[[PTRS:.*]] = getelementptr inbounds float, float* %a, <vscale x 4 x i64> %[[IND]]
; CHECK:   %[[GLOAD:.*]] = call <vscale x 4 x float> @llvm.masked.gather.nxv4f32.nxv4p0f32(<vscale x 4 x float*> %[[PTRS]]
; CHECK:   store <vscale x 4 x float> %[[GLOAD]], <vscale x 4 x float>*
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i64, i64* %b, i64 %indvars.iv
  %0 = load i64, i64* %arrayidx, align 8
  %arrayidx3 = getelementptr inbounds float, float* %a, i64 %0
  %1 = load float, float* %arrayidx3, align 4
  %arrayidx5 = getelementptr inbounds float, float* %c, i64 %indvars.iv
  store float %1, float* %arrayidx5, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !0

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void
}

; NOTE: I deliberately chose '%b' as an array of i32 indices, since the
; additional 'sext' in the for.body loop exposes additional code paths
; during vectorisation.
define void @scatter_nxv4i32_ind32(float* noalias nocapture %a, i32* noalias nocapture readonly %b, float* noalias nocapture readonly %c, i64 %n) {
; CHECK-LABEL: @scatter_nxv4i32_ind32
; CHECK: vector.body:
; CHECK:   %[[VALS:.*]] = load <vscale x 4 x float>
; CHECK:   %[[IND:.*]] = load <vscale x 4 x i32>, <vscale x 4 x i32>* %7, align 4
; CHECK:   %[[EXTIND:.*]] = sext <vscale x 4 x i32> %[[IND]] to <vscale x 4 x i64>
; CHECK:   %[[PTRS:.*]] = getelementptr inbounds float, float* %a, <vscale x 4 x i64> %[[EXTIND]]
; CHECK:   call void @llvm.masked.scatter.nxv4f32.nxv4p0f32(<vscale x 4 x float> %[[VALS]], <vscale x 4 x float*> %[[PTRS]]
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %c, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
  %1 = load i32, i32* %arrayidx3, align 4
  %idxprom4 = sext i32 %1 to i64
  %arrayidx5 = getelementptr inbounds float, float* %a, i64 %idxprom4
  store float %0, float* %arrayidx5, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !0

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void
}

define void @scatter_inv_nxv4i32(i32* noalias nocapture %inv, i32* noalias nocapture readonly %b, i64 %n) {
; CHECK-LABEL: @scatter_inv_nxv4i32
; CHECK: vector.ph:
; CHECK:   %[[INS:.*]] = insertelement <vscale x 4 x i32*> poison, i32* %inv, i32 0
; CHECK:   %[[PTRSPLAT:.*]] = shufflevector <vscale x 4 x i32*> %[[INS]], <vscale x 4 x i32*> poison, <vscale x 4 x i32> zeroinitializer
; CHECK: vector.body:
; CHECK:   %[[VALS:.*]] = load <vscale x 4 x i32>, <vscale x 4 x i32>* %5, align 4
; CHECK:   %[[MASK:.*]] = icmp ne <vscale x 4 x i32> %[[VALS]],
; CHECK:   call void @llvm.masked.scatter.nxv4i32.nxv4p0i32({{.*}}, <vscale x 4 x i32*> %[[PTRSPLAT]], i32 4, <vscale x 4 x i1> %[[MASK]])
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.inc
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %tobool.not = icmp eq i32 %0, 0
  br i1 %tobool.not, label %for.inc, label %if.then

if.then:                                          ; preds = %for.body
  store i32 3, i32* %inv, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !0

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void
}

define void @gather_inv_nxv4i32(i32* noalias nocapture %a, i32* noalias nocapture readonly %inv, i64 %n) {
; CHECK-LABEL: @gather_inv_nxv4i32
; CHECK: vector.ph:
; CHECK:   %[[INS:.*]] = insertelement <vscale x 4 x i32*> poison, i32* %inv, i32 0
; CHECK:   %[[PTRSPLAT:.*]] = shufflevector <vscale x 4 x i32*> %[[INS]], <vscale x 4 x i32*> poison, <vscale x 4 x i32> zeroinitializer
; CHECK: vector.body:
; CHECK:   %[[VALS:.*]] = load <vscale x 4 x i32>, <vscale x 4 x i32>* %5, align 4
; CHECK:   %[[MASK:.*]] = icmp sgt <vscale x 4 x i32> %[[VALS]],
; CHECK:   %{{.*}} = call <vscale x 4 x i32> @llvm.masked.gather.nxv4i32.nxv4p0i32(<vscale x 4 x i32*> %[[PTRSPLAT]], i32 4, <vscale x 4 x i1> %[[MASK]]
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.inc
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %cmp2 = icmp sgt i32 %0, 3
  br i1 %cmp2, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %1 = load i32, i32* %inv, align 4
  store i32 %1, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !0

for.cond.cleanup:                                 ; preds = %for.inc, %entry
  ret void
}

!0 = distinct !{!0, !1, !2, !3, !4, !5}
!1 = !{!"llvm.loop.mustprogress"}
!2 = !{!"llvm.loop.vectorize.width", i32 4}
!3 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
!4 = !{!"llvm.loop.interleave.count", i32 1}
!5 = !{!"llvm.loop.vectorize.enable", i1 true}
