; RUN: opt -mtriple=armv8m.main-none-none-eabi -mattr=+dsp < %s -arm-parallel-dsp -S | FileCheck %s
;
; The Cortex-M0 does not support unaligned accesses:
; RUN: opt -mtriple=armv6m-none-none-eabi < %s -arm-parallel-dsp -S | FileCheck %s --check-prefix=CHECK-UNSUPPORTED
;
; Check DSP extension:
; RUN: opt -mtriple=armv8m.main-none-none-eabi -mattr=-dsp < %s -arm-parallel-dsp -S | FileCheck %s --check-prefix=CHECK-UNSUPPORTED

define dso_local i64 @OneReduction(i32 %arg, ptr nocapture readnone %arg1, ptr nocapture readonly %arg2, ptr nocapture readonly %arg3) {
;
; CHECK-LABEL: @OneReduction
; CHECK:  %mac1{{\.}}026 = phi i64 [ [[V8:%[0-9]+]], %for.body ], [ 0, %for.body.preheader ]
; CHECK:  [[V7:%[0-9]+]] = load i32, ptr %arrayidx, align 2
; CHECK:  [[V5:%[0-9]+]] = load i32, ptr %arrayidx3, align 2
; CHECK:  [[V8]] = call i64 @llvm.arm.smlald(i32 [[V5]], i32 [[V7]], i64 %mac1{{\.}}026)
; CHECK-NOT: call i64 @llvm.arm.smlald
;
; CHECK-UNSUPPORTED-NOT:  call i64 @llvm.arm.smlald
;
entry:
  %cmp24 = icmp sgt i32 %arg, 0
  br i1 %cmp24, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  %.pre = load i16, ptr %arg3, align 2
  %.pre27 = load i16, ptr %arg2, align 2
  br label %for.body

for.cond.cleanup:
  %mac1.0.lcssa = phi i64 [ 0, %entry ], [ %add11, %for.body ]
  ret i64 %mac1.0.lcssa

for.body:
; One reduction statement here:
  %mac1.026 = phi i64 [ %add11, %for.body ], [ 0, %for.body.preheader ]

  %i.025 = phi i32 [ %add, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i16, ptr %arg3, i32 %i.025
  %0 = load i16, ptr %arrayidx, align 2
  %add = add nuw nsw i32 %i.025, 1
  %arrayidx1 = getelementptr inbounds i16, ptr %arg3, i32 %add
  %1 = load i16, ptr %arrayidx1, align 2
  %arrayidx3 = getelementptr inbounds i16, ptr %arg2, i32 %i.025
  %2 = load i16, ptr %arrayidx3, align 2
  %conv = sext i16 %2 to i32
  %conv4 = sext i16 %0 to i32
  %mul = mul nsw i32 %conv, %conv4
  %sext0 = sext i32 %mul to i64
  %arrayidx6 = getelementptr inbounds i16, ptr %arg2, i32 %add
  %3 = load i16, ptr %arrayidx6, align 2
  %conv7 = sext i16 %3 to i32
  %conv8 = sext i16 %1 to i32
  %mul9 = mul nsw i32 %conv7, %conv8
  %sext1 = sext i32 %mul9 to i64
  %add10 = add i64 %sext0, %mac1.026

; Here the Mul is the LHS, and the Add the RHS.
  %add11 = add i64 %sext1, %add10

  %exitcond = icmp ne i32 %add, %arg
  br i1 %exitcond, label %for.body, label %for.cond.cleanup
}

define dso_local arm_aapcs_vfpcc i64 @TwoReductions(i32 %arg, ptr nocapture readnone %arg1, ptr nocapture readonly %arg2, ptr nocapture readonly %arg3) {
;
; CHECK-LABEL: @TwoReductions
;
; CHECK:  %mac1{{\.}}058 = phi i64 [ [[V10:%[0-9]+]], %for.body ], [ 0, %for.body.preheader ]
; CHECK:  %mac2{{\.}}057 = phi i64 [ [[V17:%[0-9]+]], %for.body ], [ 0, %for.body.preheader ]
; CHECK:  [[V10]] = call i64 @llvm.arm.smlald(i32 %{{.*}}, i32 %{{.*}}, i64 %mac1{{\.}}058)
; CHECK:  [[V17]] = call i64 @llvm.arm.smlald(i32 %{{.*}}, i32 %{{.*}}, i64 %mac2{{\.}}057)
; CHECK-NOT: call i64 @llvm.arm.smlald
;
entry:
  %cmp55 = icmp sgt i32 %arg, 0
  br i1 %cmp55, label %for.body.preheader, label %for.cond.cleanup

for.cond.cleanup:
  %mac2.0.lcssa = phi i64 [ 0, %entry ], [ %add28, %for.body ]
  %mac1.0.lcssa = phi i64 [ 0, %entry ], [ %add16, %for.body ]
  %add30 = add nsw i64 %mac1.0.lcssa, %mac2.0.lcssa
  ret i64 %add30

for.body.preheader:
  br label %for.body

for.body:
; And two reduction statements here:
  %mac1.058 = phi i64 [ %add16, %for.body ], [ 0, %for.body.preheader ]
  %mac2.057 = phi i64 [ %add28, %for.body ], [ 0, %for.body.preheader ]

  %i.056 = phi i32 [ %add29, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i16, ptr %arg3, i32 %i.056
  %0 = load i16, ptr %arrayidx, align 2
  %add1 = or disjoint i32 %i.056, 1
  %arrayidx2 = getelementptr inbounds i16, ptr %arg3, i32 %add1
  %1 = load i16, ptr %arrayidx2, align 2
  %add3 = or disjoint i32 %i.056, 2
  %arrayidx4 = getelementptr inbounds i16, ptr %arg3, i32 %add3
  %2 = load i16, ptr %arrayidx4, align 2

  %add5 = or disjoint i32 %i.056, 3
  %arrayidx6 = getelementptr inbounds i16, ptr %arg3, i32 %add5
  %3 = load i16, ptr %arrayidx6, align 2
  %arrayidx8 = getelementptr inbounds i16, ptr %arg2, i32 %i.056
  %4 = load i16, ptr %arrayidx8, align 2
  %conv = sext i16 %4 to i32
  %conv9 = sext i16 %0 to i32
  %mul = mul nsw i32 %conv, %conv9
  %sext0 = sext i32 %mul to i64
  %arrayidx11 = getelementptr inbounds i16, ptr %arg2, i32 %add1
  %5 = load i16, ptr %arrayidx11, align 2
  %conv12 = sext i16 %5 to i32
  %conv13 = sext i16 %1 to i32
  %mul14 = mul nsw i32 %conv12, %conv13
  %sext1 = sext i32 %mul14 to i64
  %add15 = add i64 %sext0, %mac1.058
  %add16 = add i64 %add15, %sext1
  %arrayidx18 = getelementptr inbounds i16, ptr %arg2, i32 %add3
  %6 = load i16, ptr %arrayidx18, align 2
  %conv19 = sext i16 %6 to i32
  %conv20 = sext i16 %2 to i32
  %mul21 = mul nsw i32 %conv19, %conv20
  %sext2 = sext i32 %mul21 to i64
  %arrayidx23 = getelementptr inbounds i16, ptr %arg2, i32 %add5
  %7 = load i16, ptr %arrayidx23, align 2
  %conv24 = sext i16 %7 to i32
  %conv25 = sext i16 %3 to i32
  %mul26 = mul nsw i32 %conv24, %conv25
  %sext3 = sext i32 %mul26 to i64
  %add27 = add i64 %sext2, %mac2.057
  %add28 = add i64 %add27, %sext3
  %add29 = add nuw nsw i32 %i.056, 4
  %cmp = icmp slt i32 %add29, %arg
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

define i64 @zext_mul_reduction(i32 %arg, ptr nocapture readnone %arg1, ptr nocapture readonly %arg2, ptr nocapture readonly %arg3) {
; CHECK-LABEL: @zext_mul_reduction
; CHECK-NOT: call i64 @llvm.arm.smlald
; CHECK-NOT: call i32 @llvm.arm.smlad
entry:
  %cmp24 = icmp sgt i32 %arg, 0
  br i1 %cmp24, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  %.pre = load i16, ptr %arg3, align 2
  %.pre27 = load i16, ptr %arg2, align 2
  br label %for.body

for.cond.cleanup:
  %mac1.0.lcssa = phi i64 [ 0, %entry ], [ %add11, %for.body ]
  ret i64 %mac1.0.lcssa

for.body:
  %mac1.026 = phi i64 [ %add11, %for.body ], [ 0, %for.body.preheader ]
  %i.025 = phi i32 [ %add, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i16, ptr %arg3, i32 %i.025
  %0 = load i16, ptr %arrayidx, align 2
  %add = add nuw nsw i32 %i.025, 1
  %arrayidx1 = getelementptr inbounds i16, ptr %arg3, i32 %add
  %1 = load i16, ptr %arrayidx1, align 2
  %arrayidx3 = getelementptr inbounds i16, ptr %arg2, i32 %i.025
  %2 = load i16, ptr %arrayidx3, align 2
  %conv = zext i16 %2 to i32
  %conv4 = sext i16 %0 to i32
  %mul = mul nsw i32 %conv, %conv4
  %sext0 = sext i32 %mul to i64
  %arrayidx6 = getelementptr inbounds i16, ptr %arg2, i32 %add
  %3 = load i16, ptr %arrayidx6, align 2
  %conv7 = zext i16 %3 to i32
  %conv8 = sext i16 %1 to i32
  %mul9 = mul nsw i32 %conv7, %conv8
  %sext1 = sext i32 %mul9 to i64
  %add10 = add i64 %sext0, %mac1.026
  %add11 = add i64 %sext1, %add10
  %exitcond = icmp ne i32 %add, %arg
  br i1 %exitcond, label %for.body, label %for.cond.cleanup
}

define i64 @zext_add_reduction(i32 %arg, ptr nocapture readnone %arg1, ptr nocapture readonly %arg2, ptr nocapture readonly %arg3) {
; CHECK-LABEL: @zext_add_reduction
; CHECK-NOT: call i64 @llvm.arm.smlald
; CHECK-NOT: call i32 @llvm.arm.smlad
entry:
  %cmp24 = icmp sgt i32 %arg, 0
  br i1 %cmp24, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  %.pre = load i16, ptr %arg3, align 2
  %.pre27 = load i16, ptr %arg2, align 2
  br label %for.body

for.cond.cleanup:
  %mac1.0.lcssa = phi i64 [ 0, %entry ], [ %add11, %for.body ]
  ret i64 %mac1.0.lcssa

for.body:
  %mac1.026 = phi i64 [ %add11, %for.body ], [ 0, %for.body.preheader ]
  %i.025 = phi i32 [ %add, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i16, ptr %arg3, i32 %i.025
  %0 = load i16, ptr %arrayidx, align 2
  %add = add nuw nsw i32 %i.025, 1
  %arrayidx1 = getelementptr inbounds i16, ptr %arg3, i32 %add
  %1 = load i16, ptr %arrayidx1, align 2
  %arrayidx3 = getelementptr inbounds i16, ptr %arg2, i32 %i.025
  %2 = load i16, ptr %arrayidx3, align 2
  %conv = sext i16 %2 to i32
  %conv4 = sext i16 %0 to i32
  %mul = mul nsw i32 %conv, %conv4
  %sext0 = zext i32 %mul to i64
  %arrayidx6 = getelementptr inbounds i16, ptr %arg2, i32 %add
  %3 = load i16, ptr %arrayidx6, align 2
  %conv7 = sext i16 %3 to i32
  %conv8 = sext i16 %1 to i32
  %mul9 = mul nsw i32 %conv7, %conv8
  %sext1 = zext i32 %mul9 to i64
  %add10 = add i64 %sext0, %mac1.026
  %add11 = add i64 %sext1, %add10
  %exitcond = icmp ne i32 %add, %arg
  br i1 %exitcond, label %for.body, label %for.cond.cleanup
}
