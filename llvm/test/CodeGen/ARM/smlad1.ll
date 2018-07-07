; RUN: opt -mtriple=arm-arm-eabi -mcpu=cortex-m33 < %s -arm-parallel-dsp -S | FileCheck %s

; CHECK-LABEL: @test1
; CHECK:  %mac1{{\.}}026 = phi i32 [ [[V8:%[0-9]+]], %for.body ], [ 0, %for.body.preheader ]
; CHECK:  [[V4:%[0-9]+]] = bitcast i16* %arrayidx3 to i32*
; CHECK:  [[V5:%[0-9]+]] = load i32, i32* [[V4]], align 2
; CHECK:  [[V6:%[0-9]+]] = bitcast i16* %arrayidx to i32*
; CHECK:  [[V7:%[0-9]+]] = load i32, i32* [[V6]], align 2
; CHECK:  [[V8]] = call i32 @llvm.arm.smlad(i32 [[V5]], i32 [[V7]], i32 %mac1{{\.}}026)

define dso_local i32 @test1(i32 %arg, i32* nocapture readnone %arg1, i16* nocapture readonly %arg2, i16* nocapture readonly %arg3) {
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
  %conv4 = sext i16 %0 to i32
  %mul = mul nsw i32 %conv, %conv4
  %arrayidx6 = getelementptr inbounds i16, i16* %arg2, i32 %add
  %3 = load i16, i16* %arrayidx6, align 2
  %conv7 = sext i16 %3 to i32
  %conv8 = sext i16 %1 to i32
  %mul9 = mul nsw i32 %conv7, %conv8
  %add10 = add i32 %mul, %mac1.026

; And here the Add is the LHS, the Mul the RHS
  %add11 = add i32 %add10, %mul9

  %exitcond = icmp ne i32 %add, %arg
  br i1 %exitcond, label %for.body, label %for.cond.cleanup
}

; Here we have i8 loads, which we do want to support, but don't handle yet.
;
; CHECK-LABEL: @test2
; CHECK-NOT:   call i32 @llvm.arm.smlad
;
define dso_local i32 @test2(i32 %arg, i32* nocapture readnone %arg1, i8* nocapture readonly %arg2, i8* nocapture readonly %arg3) {
entry:
  %cmp24 = icmp sgt i32 %arg, 0
  br i1 %cmp24, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  %.pre = load i8, i8* %arg3, align 2
  %.pre27 = load i8, i8* %arg2, align 2
  br label %for.body

for.cond.cleanup:
  %mac1.0.lcssa = phi i32 [ 0, %entry ], [ %add11, %for.body ]
  ret i32 %mac1.0.lcssa

for.body:
  %mac1.026 = phi i32 [ %add11, %for.body ], [ 0, %for.body.preheader ]
  %i.025 = phi i32 [ %add, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i8, i8* %arg3, i32 %i.025
  %0 = load i8, i8* %arrayidx, align 2
  %add = add nuw nsw i32 %i.025, 1
  %arrayidx1 = getelementptr inbounds i8, i8* %arg3, i32 %add
  %1 = load i8, i8* %arrayidx1, align 2
  %arrayidx3 = getelementptr inbounds i8, i8* %arg2, i32 %i.025
  %2 = load i8, i8* %arrayidx3, align 2
  %conv = sext i8 %2 to i32
  %conv4 = sext i8 %0 to i32
  %mul = mul nsw i32 %conv, %conv4
  %arrayidx6 = getelementptr inbounds i8, i8* %arg2, i32 %add
  %3 = load i8, i8* %arrayidx6, align 2
  %conv7 = sext i8 %3 to i32
  %conv8 = sext i8 %1 to i32
  %mul9 = mul nsw i32 %conv7, %conv8
  %add10 = add i32 %mul, %mac1.026
  %add11 = add i32 %add10, %mul9
  %exitcond = icmp ne i32 %add, %arg
  br i1 %exitcond, label %for.body, label %for.cond.cleanup
}


