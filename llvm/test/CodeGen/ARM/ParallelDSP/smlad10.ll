; RUN: opt -mtriple=armv8m.main-none-none-eabi -mattr=+dsp < %s -arm-parallel-dsp -S | FileCheck %s
;
; Reduction statement is an i64 type: we only support i32 so check that the
; rewrite isn't triggered.
;
; CHECK-NOT:  call i32 @llvm.arm.smlad
;
define dso_local i64 @test(i64 %arg, ptr nocapture readnone %arg1, ptr nocapture readonly %arg2, ptr nocapture readonly %arg3) {
entry:
  %cmp24 = icmp sgt i64 %arg, 0
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
  %i.025 = phi i64 [ %add, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i16, ptr %arg3, i64 %i.025
  %0 = load i16, ptr %arrayidx, align 2
  %add = add nuw nsw i64 %i.025, 1
  %arrayidx1 = getelementptr inbounds i16, ptr %arg3, i64 %add
  %1 = load i16, ptr %arrayidx1, align 2
  %arrayidx3 = getelementptr inbounds i16, ptr %arg2, i64 %i.025
  %2 = load i16, ptr %arrayidx3, align 2
  %conv = sext i16 %2 to i64
  %conv4 = sext i16 %0 to i64
  %mul = mul nsw i64 %conv, %conv4
  %arrayidx6 = getelementptr inbounds i16, ptr %arg2, i64 %add
  %3 = load i16, ptr %arrayidx6, align 2
  %conv7 = sext i16 %3 to i64
  %conv8 = sext i16 %1 to i64
  %mul9 = mul nsw i64 %conv7, %conv8
  %add10 = add i64 %mul, %mac1.026

  %add11 = add i64 %mul9, %add10

  %exitcond = icmp ne i64 %add, %arg
  br i1 %exitcond, label %for.body, label %for.cond.cleanup
}

