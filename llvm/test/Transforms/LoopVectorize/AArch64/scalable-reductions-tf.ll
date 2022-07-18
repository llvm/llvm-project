; RUN: opt < %s -loop-vectorize -prefer-predicate-over-epilogue=predicate-else-scalar-epilogue \
; RUN:   -mtriple aarch64-unknown-linux-gnu -mattr=+sve -S | FileCheck %s

define void @invariant_store_red_exit_is_phi(i32* %dst, i32* readonly %src, i64 %n) {
; CHECK-LABEL: @invariant_store_red_exit_is_phi(
; CHECK: vector.ph:
; CHECK:      %[[ACTIVE_LANE_MASK_ENTRY:.*]] = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i64(i64 0, i64 %n)
; CHECK: vector.body:
; CHECK:      %[[ACTIVE_LANE_MASK:.*]] = phi <vscale x 4 x i1> [ %[[ACTIVE_LANE_MASK_ENTRY]], %vector.ph ], [ %[[ACTIVE_LANE_MASK_NEXT:.*]], %vector.body ]
; CHECK:      %[[VEC_PHI:.*]] = phi <vscale x 4 x i32> [ zeroinitializer, %vector.ph ], [ %[[PREDPHI:.*]], %vector.body ]
; CHECK:      %[[LOAD:.*]] = call <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0nxv4i32
; CHECK-NEXT: %[[ADD:.*]] = add <vscale x 4 x i32> %[[VEC_PHI]], %[[LOAD]]
; CHECK-NEXT: %[[SELECT:.*]] = select <vscale x 4 x i1> %[[ACTIVE_LANE_MASK]], <vscale x 4 x i32> %[[ADD]], <vscale x 4 x i32> %[[VEC_PHI]]
; CHECK:      %[[ACTIVE_LANE_MASK_NEXT]] = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i64(i64 %{{.*}}, i64 %n)
; CHECK: middle.block:
; CHECK-NEXT: %[[SUM:.*]] = call i32 @llvm.vector.reduce.add.nxv4i32(<vscale x 4 x i32> %[[SELECT]])
; CHECK-NEXT: store i32 %[[SUM]], i32* %dst, align 4
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.inc
  %red = phi i32 [ 0, %entry ], [ %storemerge, %for.body ]
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx6 = getelementptr inbounds i32, i32* %src, i64 %indvars.iv
  %load = load i32, i32* %arrayidx6, align 4
  %storemerge = add i32 %red, %load
  store i32 %storemerge, i32* %dst, align 4
  %indvars.iv.next = add nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %for.end.loopexit, label %for.body, !llvm.loop !0

for.end.loopexit:                                 ; preds = %for.inc
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit
  ret void
}

!0 = distinct !{!0, !1, !2, !3, !4}
!1 = !{!"llvm.loop.vectorize.width", i32 4}
!2 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
!3 = !{!"llvm.loop.interleave.count", i32 1}
!4 = !{!"llvm.loop.vectorize.enable", i1 true}
