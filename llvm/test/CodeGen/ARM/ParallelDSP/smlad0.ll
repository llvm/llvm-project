; RUN: opt -mtriple=arm-none-none-eabi -mcpu=cortex-m33 < %s -arm-parallel-dsp -S | FileCheck %s
; RUN: opt -mtriple=armeb-arm-eabi -mcpu=cortex-m0 < %s -arm-parallel-dsp -S | FileCheck %s --check-prefix=CHECK-UNSUPPORTED
;
; The Cortex-M0 does not support unaligned accesses:
; RUN: opt -mtriple=arm-none-none-eabi -mcpu=cortex-m0 < %s -arm-parallel-dsp -S | FileCheck %s --check-prefix=CHECK-UNSUPPORTED
;
; Check DSP extension:
; RUN: opt -mtriple=arm-none-none-eabi -mcpu=cortex-m33 -mattr=-dsp < %s -arm-parallel-dsp -S | FileCheck %s --check-prefix=CHECK-UNSUPPORTED

define dso_local i32 @OneReduction(i32 %arg, i32* nocapture readnone %arg1, i16* nocapture readonly %arg2, i16* nocapture readonly %arg3) {
;
; CHECK-LABEL: @OneReduction
; CHECK:  %mac1{{\.}}026 = phi i32 [ [[V8:%[0-9]+]], %for.body ], [ 0, %for.body.preheader ]
; CHECK:  [[V4:%[0-9]+]] = bitcast i16* %arrayidx to i32*
; CHECK:  [[V5:%[0-9]+]] = load i32, i32* [[V4]], align 2
; CHECK:  [[V6:%[0-9]+]] = bitcast i16* %arrayidx3 to i32*
; CHECK:  [[V7:%[0-9]+]] = load i32, i32* [[V6]], align 2
; CHECK:  [[V8]] = call i32 @llvm.arm.smlad(i32 [[V7]], i32 [[V5]], i32 %mac1{{\.}}026)
; CHECK-NOT: call i32 @llvm.arm.smlad
;
; CHECK-UNSUPPORTED-NOT:  call i32 @llvm.arm.smlad
;
entry:
  %cmp24 = icmp sgt i32 %arg, 0
  br i1 %cmp24, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  %.pre = load i16, i16* %arg3, align 2
  %.pre27 = load i16, i16* %arg2, align 2
  br label %for.body

for.cond.cleanup:
  %mac1.0.lcssa = phi i32 [ 0, %entry ], [ %add11, %for.body ]
  ret i32 %mac1.0.lcssa

for.body:
; One reduction statement here:
  %mac1.026 = phi i32 [ %add11, %for.body ], [ 0, %for.body.preheader ]

  %i.025 = phi i32 [ %add, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i16, i16* %arg3, i32 %i.025
  %0 = load i16, i16* %arrayidx, align 2
  %add = add nuw nsw i32 %i.025, 1
  %arrayidx1 = getelementptr inbounds i16, i16* %arg3, i32 %add
  %1 = load i16, i16* %arrayidx1, align 2
  %arrayidx3 = getelementptr inbounds i16, i16* %arg2, i32 %i.025
  %2 = load i16, i16* %arrayidx3, align 2
  %conv = sext i16 %2 to i32
  %conv4 = sext i16 %0 to i32
  %mul = mul nsw i32 %conv, %conv4
  %arrayidx6 = getelementptr inbounds i16, i16* %arg2, i32 %add
  %3 = load i16, i16* %arrayidx6, align 2
  %conv7 = sext i16 %3 to i32
  %conv8 = sext i16 %1 to i32
  %mul9 = mul nsw i32 %conv7, %conv8
  %add10 = add i32 %mul, %mac1.026

; Here the Mul is the LHS, and the Add the RHS.
  %add11 = add i32 %mul9, %add10

  %exitcond = icmp ne i32 %add, %arg
  br i1 %exitcond, label %for.body, label %for.cond.cleanup
}

define dso_local arm_aapcs_vfpcc i32 @TwoReductions(i32 %arg, i32* nocapture readnone %arg1, i16* nocapture readonly %arg2, i16* nocapture readonly %arg3) {
;
; CHECK-LABEL: @TwoReductions
;
; CHECK:  %mac1{{\.}}058 = phi i32 [ [[V10:%[0-9]+]], %for.body ], [ 0, %for.body.preheader ]
; CHECK:  %mac2{{\.}}057 = phi i32 [ [[V17:%[0-9]+]], %for.body ], [ 0, %for.body.preheader ]
; CHECK:  [[V10]] = call i32 @llvm.arm.smlad(i32 %{{.*}}, i32 %{{.*}}, i32 %mac1{{\.}}058)
; CHECK:  [[V17]] = call i32 @llvm.arm.smlad(i32 %{{.*}}, i32 %{{.*}}, i32 %mac2{{\.}}057)
; CHECK-NOT: call i32 @llvm.arm.smlad
;
entry:
  %cmp55 = icmp sgt i32 %arg, 0
  br i1 %cmp55, label %for.body.preheader, label %for.cond.cleanup

for.cond.cleanup:
  %mac2.0.lcssa = phi i32 [ 0, %entry ], [ %add28, %for.body ]
  %mac1.0.lcssa = phi i32 [ 0, %entry ], [ %add16, %for.body ]
  %add30 = add nsw i32 %mac1.0.lcssa, %mac2.0.lcssa
  ret i32 %add30

for.body.preheader:
  br label %for.body

for.body:
; And two reduction statements here:
  %mac1.058 = phi i32 [ %add16, %for.body ], [ 0, %for.body.preheader ]
  %mac2.057 = phi i32 [ %add28, %for.body ], [ 0, %for.body.preheader ]

  %i.056 = phi i32 [ %add29, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i16, i16* %arg3, i32 %i.056
  %0 = load i16, i16* %arrayidx, align 2
  %add1 = or i32 %i.056, 1
  %arrayidx2 = getelementptr inbounds i16, i16* %arg3, i32 %add1
  %1 = load i16, i16* %arrayidx2, align 2
  %add3 = or i32 %i.056, 2
  %arrayidx4 = getelementptr inbounds i16, i16* %arg3, i32 %add3
  %2 = load i16, i16* %arrayidx4, align 2

  %add5 = or i32 %i.056, 3
  %arrayidx6 = getelementptr inbounds i16, i16* %arg3, i32 %add5
  %3 = load i16, i16* %arrayidx6, align 2
  %arrayidx8 = getelementptr inbounds i16, i16* %arg2, i32 %i.056
  %4 = load i16, i16* %arrayidx8, align 2
  %conv = sext i16 %4 to i32
  %conv9 = sext i16 %0 to i32
  %mul = mul nsw i32 %conv, %conv9
  %arrayidx11 = getelementptr inbounds i16, i16* %arg2, i32 %add1
  %5 = load i16, i16* %arrayidx11, align 2
  %conv12 = sext i16 %5 to i32
  %conv13 = sext i16 %1 to i32
  %mul14 = mul nsw i32 %conv12, %conv13
  %add15 = add i32 %mul, %mac1.058
  %add16 = add i32 %add15, %mul14
  %arrayidx18 = getelementptr inbounds i16, i16* %arg2, i32 %add3
  %6 = load i16, i16* %arrayidx18, align 2
  %conv19 = sext i16 %6 to i32
  %conv20 = sext i16 %2 to i32
  %mul21 = mul nsw i32 %conv19, %conv20
  %arrayidx23 = getelementptr inbounds i16, i16* %arg2, i32 %add5
  %7 = load i16, i16* %arrayidx23, align 2
  %conv24 = sext i16 %7 to i32
  %conv25 = sext i16 %3 to i32
  %mul26 = mul nsw i32 %conv24, %conv25
  %add27 = add i32 %mul21, %mac2.057
  %add28 = add i32 %add27, %mul26
  %add29 = add nuw nsw i32 %i.056, 4
  %cmp = icmp slt i32 %add29, %arg
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

define i32 @one_zext(i32 %arg, i32* nocapture readnone %arg1, i16* nocapture readonly %arg2, i16* nocapture readonly %arg3) {
; CHECK-LABEL: @one_zext
; CHECK-NOT: call i32 @llvm.arm.smlad
entry:
  %cmp24 = icmp sgt i32 %arg, 0
  br i1 %cmp24, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  %.pre = load i16, i16* %arg3, align 2
  %.pre27 = load i16, i16* %arg2, align 2
  br label %for.body

for.cond.cleanup:
  %mac1.0.lcssa = phi i32 [ 0, %entry ], [ %add11, %for.body ]
  ret i32 %mac1.0.lcssa

for.body:
  %mac1.026 = phi i32 [ %add11, %for.body ], [ 0, %for.body.preheader ]
  %i.025 = phi i32 [ %add, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i16, i16* %arg3, i32 %i.025
  %0 = load i16, i16* %arrayidx, align 2
  %add = add nuw nsw i32 %i.025, 1
  %arrayidx1 = getelementptr inbounds i16, i16* %arg3, i32 %add
  %1 = load i16, i16* %arrayidx1, align 2
  %arrayidx3 = getelementptr inbounds i16, i16* %arg2, i32 %i.025
  %2 = load i16, i16* %arrayidx3, align 2
  %conv = sext i16 %2 to i32
  %conv4 = zext i16 %0 to i32
  %mul = mul nsw i32 %conv, %conv4
  %arrayidx6 = getelementptr inbounds i16, i16* %arg2, i32 %add
  %3 = load i16, i16* %arrayidx6, align 2
  %conv7 = sext i16 %3 to i32
  %conv8 = zext i16 %1 to i32
  %mul9 = mul nsw i32 %conv7, %conv8
  %add10 = add i32 %mul, %mac1.026
  %add11 = add i32 %mul9, %add10
  %exitcond = icmp ne i32 %add, %arg
  br i1 %exitcond, label %for.body, label %for.cond.cleanup
}

define i32 @two_zext(i32 %arg, i32* nocapture readnone %arg1, i16* nocapture readonly %arg2, i16* nocapture readonly %arg3) {
; CHECK-LABEL: @two_zext
; CHECK-NOT: call i32 @llvm.arm.smlad
entry:
  %cmp24 = icmp sgt i32 %arg, 0
  br i1 %cmp24, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  %.pre = load i16, i16* %arg3, align 2
  %.pre27 = load i16, i16* %arg2, align 2
  br label %for.body

for.cond.cleanup:
  %mac1.0.lcssa = phi i32 [ 0, %entry ], [ %add11, %for.body ]
  ret i32 %mac1.0.lcssa

for.body:
  %mac1.026 = phi i32 [ %add11, %for.body ], [ 0, %for.body.preheader ]
  %i.025 = phi i32 [ %add, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i16, i16* %arg3, i32 %i.025
  %0 = load i16, i16* %arrayidx, align 2
  %add = add nuw nsw i32 %i.025, 1
  %arrayidx1 = getelementptr inbounds i16, i16* %arg3, i32 %add
  %1 = load i16, i16* %arrayidx1, align 2
  %arrayidx3 = getelementptr inbounds i16, i16* %arg2, i32 %i.025
  %2 = load i16, i16* %arrayidx3, align 2
  %conv = zext i16 %2 to i32
  %conv4 = zext i16 %0 to i32
  %mul = mul nsw i32 %conv, %conv4
  %arrayidx6 = getelementptr inbounds i16, i16* %arg2, i32 %add
  %3 = load i16, i16* %arrayidx6, align 2
  %conv7 = zext i16 %3 to i32
  %conv8 = zext i16 %1 to i32
  %mul9 = mul nsw i32 %conv7, %conv8
  %add10 = add i32 %mul, %mac1.026
  %add11 = add i32 %mul9, %add10
  %exitcond = icmp ne i32 %add, %arg
  br i1 %exitcond, label %for.body, label %for.cond.cleanup
}

