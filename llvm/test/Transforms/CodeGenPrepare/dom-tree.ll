; REQUIRES: arm-registered-target
; RUN: opt -S -loop-unroll -codegenprepare < %s -domtree -verify-dom-info | FileCheck %s
;
; Checks that the dom tree is properly invalidated after an operation that will
; invalidate it in CodeGenPrepare. If the domtree isn't properly invalidated,
; this will likely segfault, or print badref.

; CHECK: @f

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7--linux-gnueabihf"

define i32 @f(i32 %a) #0 {
entry:
  br label %for.body

for.cond.cleanup:
  ret i32 %or

for.body:
  %i.08 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %b.07 = phi i32 [ 0, %entry ], [ %or, %for.body ]
  %shr = lshr i32 %a, %i.08
  %and = and i32 %shr, 1
  %sub = sub nuw nsw i32 31, %i.08
  %shl = shl i32 %and, %sub
  %or = or i32 %shl, %b.07
  %inc = add nuw nsw i32 %i.08, 1
  %exitcond = icmp eq i32 %inc, 32
  br i1 %exitcond, label %for.cond.cleanup, label %for.body, !llvm.loop !3
}

attributes #0 = { norecurse nounwind readnone "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="cortex-a8" "target-features"="+dsp,+neon,+vfp3" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"min_enum_size", i32 4}
!2 = !{!"clang version 3.8.0"}
!3 = distinct !{!3, !4}
!4 = !{!"llvm.loop.unroll.full"}
