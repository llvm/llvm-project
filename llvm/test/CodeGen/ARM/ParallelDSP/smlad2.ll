; RUN: opt -mtriple=armv8m.main-none-none-eabi -mattr=+dsp < %s -arm-parallel-dsp -S | FileCheck %s
;
; Operands of both muls are not symmetrical (see also comments inlined below), check
; that the rewrite isn't triggered.
;
; CHECK-NOT:  call i32 @llvm.arm.smlad
;
define dso_local i32 @test(i32 %arg, ptr nocapture readnone %arg1, ptr nocapture readonly %arg2, ptr nocapture readonly %arg3) {
entry:
  %cmp24 = icmp sgt i32 %arg, 0
  br i1 %cmp24, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  %.pre = load i16, ptr %arg3, align 2
  %.pre27 = load i16, ptr %arg2, align 2
  br label %for.body

for.cond.cleanup:
  %mac1.0.lcssa = phi i32 [ 0, %entry ], [ %add11, %for.body ]
  ret i32 %mac1.0.lcssa

for.body:
  %mac1.026 = phi i32 [ %add11, %for.body ], [ 0, %for.body.preheader ]
  %i.025 = phi i32 [ %add, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i16, ptr %arg3, i32 %i.025
  %0 = load i16, ptr %arrayidx, align 2
  %add = add nuw nsw i32 %i.025, 1
  %arrayidx1 = getelementptr inbounds i16, ptr %arg3, i32 %add
  %1 = load i16, ptr %arrayidx1, align 2
  %arrayidx3 = getelementptr inbounds i16, ptr %arg2, i32 %i.025
  %2 = load i16, ptr %arrayidx3, align 2
  %conv = sext i16 %2 to i32

; This zero-extends the 2nd operand of %mul:
  %conv4 = zext i16 %0 to i32

  %mul = mul nsw i32 %conv, %conv4
  %arrayidx6 = getelementptr inbounds i16, ptr %arg2, i32 %add
  %3 = load i16, ptr %arrayidx6, align 2

; And here we only have sign-extensions. Thus, the operands of
; %mul and %mul9 are not symmetrical:
  %conv7 = sext i16 %3 to i32
  %conv8 = sext i16 %1 to i32

  %mul9 = mul nsw i32 %conv7, %conv8
  %add10 = add i32 %mul, %mac1.026
  %add11 = add i32 %add10, %mul9
  %exitcond = icmp ne i32 %add, %arg
  br i1 %exitcond, label %for.body, label %for.cond.cleanup
}
