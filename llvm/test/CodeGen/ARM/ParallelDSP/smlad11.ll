; REQUIRES: asserts
; RUN: opt -mtriple=armv8m.main-none-none-eabi -mattr=+dsp < %s -arm-parallel-dsp -S -stats 2>&1 | FileCheck %s
;
; A more complicated chain: 4 mul operations, so we expect 2 smlad calls.
;
; CHECK:  %mac1{{\.}}054 = phi i32 [ [[V17:%[0-9]+]], %for.body ], [ 0, %for.body.preheader ]
; CHECK:  [[V11:%[0-9]+]] = load i32, ptr %arrayidx, align 2
; CHECK:  [[V16:%[0-9]+]] = load i32, ptr %arrayidx4, align 2
; CHECK:  [[V9:%[0-9]+]] = load i32, ptr %arrayidx8, align 2
; CHECK:  [[ACC:%[0-9]+]] = call i32 @llvm.arm.smlad(i32 [[V9]], i32 [[V11]], i32 %mac1{{\.}}054)
; CHECK:  [[V14:%[0-9]+]] = load i32, ptr %arrayidx17, align 2
; CHECK:  [[V12:%[0-9]+]] = call i32 @llvm.arm.smlad(i32 [[V14]], i32 [[V16]], i32 [[ACC]])
;
; And we don't want to see a 3rd smlad:
; CHECK-NOT: call i32 @llvm.arm.smlad
;
; CHECK:  2 arm-parallel-dsp - Number of smlad instructions generated
;
define dso_local i32 @test(i32 %arg, ptr nocapture readnone %arg1, ptr nocapture readonly %arg2, ptr nocapture readonly %arg3) {
entry:
  %cmp52 = icmp sgt i32 %arg, 0
  br i1 %cmp52, label %for.body.preheader, label %for.cond.cleanup

for.cond.cleanup:
  %mac1.0.lcssa = phi i32 [ 0, %entry ], [ %add28, %for.body ]
  ret i32 %mac1.0.lcssa

for.body.preheader:
  br label %for.body

for.body:
  %mac1.054 = phi i32 [ %add28, %for.body ], [ 0, %for.body.preheader ]
  %i.053 = phi i32 [ %add29, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i16, ptr %arg3, i32 %i.053
  %0 = load i16, ptr %arrayidx, align 2
  %add1 = or i32 %i.053, 1
  %arrayidx2 = getelementptr inbounds i16, ptr %arg3, i32 %add1
  %1 = load i16, ptr %arrayidx2, align 2
  %add3 = or i32 %i.053, 2
  %arrayidx4 = getelementptr inbounds i16, ptr %arg3, i32 %add3
  %2 = load i16, ptr %arrayidx4, align 2
  %add5 = or i32 %i.053, 3
  %arrayidx6 = getelementptr inbounds i16, ptr %arg3, i32 %add5
  %3 = load i16, ptr %arrayidx6, align 2
  %arrayidx8 = getelementptr inbounds i16, ptr %arg2, i32 %i.053
  %4 = load i16, ptr %arrayidx8, align 2
  %conv = sext i16 %4 to i32
  %conv9 = sext i16 %0 to i32
  %mul = mul nsw i32 %conv, %conv9
  %arrayidx11 = getelementptr inbounds i16, ptr %arg2, i32 %add1
  %5 = load i16, ptr %arrayidx11, align 2
  %conv12 = sext i16 %5 to i32
  %conv13 = sext i16 %1 to i32
  %mul14 = mul nsw i32 %conv12, %conv13
  %arrayidx17 = getelementptr inbounds i16, ptr %arg2, i32 %add3
  %6 = load i16, ptr %arrayidx17, align 2
  %conv18 = sext i16 %6 to i32
  %conv19 = sext i16 %2 to i32
  %mul20 = mul nsw i32 %conv18, %conv19
  %arrayidx23 = getelementptr inbounds i16, ptr %arg2, i32 %add5
  %7 = load i16, ptr %arrayidx23, align 2
  %conv24 = sext i16 %7 to i32
  %conv25 = sext i16 %3 to i32
  %mul26 = mul nsw i32 %conv24, %conv25
  %add15 = add i32 %mul, %mac1.054
  %add21 = add i32 %add15, %mul14
  %add27 = add i32 %add21, %mul20
  %add28 = add i32 %add27, %mul26
  %add29 = add nuw nsw i32 %i.053, 4
  %cmp = icmp slt i32 %add29, %arg
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}
