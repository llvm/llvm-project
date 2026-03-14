; RUN: opt -mtriple=armv8m.main-none-none-eabi -mattr=+dsp < %s -arm-parallel-dsp -S | FileCheck %s
;
; The loads are not narrow loads: check that the rewrite isn't triggered.
;
; CHECK-NOT:  call i32 @llvm.arm.smlad
;
; Arg2 is now an i32, while Arg3 is still and i16:
;
define dso_local i32 @test(i32 %arg, ptr nocapture readnone %arg1, ptr nocapture readonly %arg2, ptr nocapture readonly %arg3) {
entry:
  %cmp22 = icmp sgt i32 %arg, 0
  br i1 %cmp22, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  %.pre = load i16, ptr %arg3, align 2
  br label %for.body

for.cond.cleanup:
  %mac1.0.lcssa = phi i32 [ 0, %entry ], [ %add9, %for.body ]
  ret i32 %mac1.0.lcssa

for.body:
  %0 = phi i16 [ %1, %for.body ], [ %.pre, %for.body.preheader ]
  %mac1.024 = phi i32 [ %add9, %for.body ], [ 0, %for.body.preheader ]
  %i.023 = phi i32 [ %add, %for.body ], [ 0, %for.body.preheader ]
  %add = add nuw nsw i32 %i.023, 1
  %arrayidx1 = getelementptr inbounds i16, ptr %arg3, i32 %add
  %1 = load i16, ptr %arrayidx1, align 2
  %conv = sext i16 %0 to i32

; This is a 'normal' i32 load to %2:
  %arrayidx3 = getelementptr inbounds i32, ptr %arg2, i32 %i.023
  %2 = load i32, ptr %arrayidx3, align 4

; This mul has now 1 operand which is a narrow load, and the other a normal
; i32 load:
  %mul = mul nsw i32 %2, %conv

  %add4 = add nuw nsw i32 %i.023, 2
  %arrayidx5 = getelementptr inbounds i32, ptr %arg2, i32 %add4
  %3 = load i32, ptr %arrayidx5, align 4
  %conv6 = sext i16 %1 to i32
  %mul7 = mul nsw i32 %3, %conv6
  %add8 = add i32 %mul, %mac1.024
  %add9 = add i32 %add8, %mul7
  %exitcond = icmp eq i32 %add, %arg
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}
