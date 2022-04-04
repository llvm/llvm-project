; RUN: opt -passes='loop-vectorize' -S -pass-remarks-missed=loop-vectorize < %s 2>&1 | FileCheck %s
;
; FP primary induction is not supported in LV. Make sure Legal bails out.
;
; CHECK: loop not vectorized

define void @PR37515() {
entry:
  br label %loop

loop:
  %p = phi float [ 19.0, %entry ], [ %a, %loop ]
  %a = fadd fast float %p, -1.0
  %m = fmul fast float %a, %a
  %c = fcmp fast ugt float %a, 2.0
  br i1 %c, label %loop, label %exit

exit:
  unreachable
}
