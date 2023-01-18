; RUN: opt -opaque-pointers=0 -passes=loop-vectorize,dce,instcombine -mtriple aarch64-linux-gnu -S < %s | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

define void @inv_store_last_lane(i32* noalias nocapture %a, i32* noalias nocapture %inv, i32* noalias nocapture readonly %b, i64 %n) #0 {
; CHECK-LABEL: @inv_store_last_lane
; CHECK: vector.body:
; CHECK:  store <vscale x 4 x i32> %[[VEC_VAL:.*]], <
; CHECK: middle.block:
; CHECK: %[[VSCALE:.*]] = call i32 @llvm.vscale.i32()
; CHECK-NEXT: %[[VSCALE2:.*]] = shl i32 %[[VSCALE]], 2
; CHECK-NEXT: %[[LAST_LANE:.*]] = add i32 %[[VSCALE2]], -1
; CHECK-NEXT: %{{.*}} = extractelement <vscale x 4 x i32> %[[VEC_VAL]], i32 %[[LAST_LANE]]

entry:
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %mul = shl nsw i32 %0, 1
  %arrayidx2 = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  store i32 %mul, i32* %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %exit, label %for.body, !llvm.loop !0

exit:              ; preds = %for.body
  %arrayidx5 = getelementptr inbounds i32, i32* %inv, i64 42
  store i32 %mul, i32* %arrayidx5, align 4
  ret void
}

define float @ret_last_lane(float* noalias nocapture %a, float* noalias nocapture readonly %b, i64 %n) #0 {
; CHECK-LABEL: @ret_last_lane
; CHECK: vector.body:
; CHECK:  store <vscale x 4 x float> %[[VEC_VAL:.*]], <
; CHECK: middle.block:
; CHECK: %[[VSCALE:.*]] = call i32 @llvm.vscale.i32()
; CHECK-NEXT: %[[VSCALE2:.*]] = shl i32 %[[VSCALE]], 2
; CHECK-NEXT: %[[LAST_LANE:.*]] = add i32 %[[VSCALE2]], -1
; CHECK-NEXT: %{{.*}} = extractelement <vscale x 4 x float> %[[VEC_VAL]], i32 %[[LAST_LANE]]

entry:
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %b, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %mul = fmul float %0, 2.000000e+00
  %arrayidx2 = getelementptr inbounds float, float* %a, i64 %indvars.iv
  store float %mul, float* %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %exit, label %for.body, !llvm.loop !6

exit:                                 ; preds = %for.body, %entry
  ret float %mul
}

attributes #0 = { "target-cpu"="generic" "target-features"="+neon,+sve" }

!0 = distinct !{!0, !1, !2, !3, !4, !5}
!1 = !{!"llvm.loop.mustprogress"}
!2 = !{!"llvm.loop.vectorize.width", i32 4}
!3 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
!4 = !{!"llvm.loop.interleave.count", i32 1}
!5 = !{!"llvm.loop.vectorize.enable", i1 true}
!6 = distinct !{!6, !1, !2, !3, !4, !5}
