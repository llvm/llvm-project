; RUN: not opt -passes=loop-unroll -S < %s 2>&1 | FileCheck %s
;
; CHECK: llvm.loop metadata must have exactly one operand.

define void @test(i1 %exitcond) {
entry:
  br label %loop
loop:
  br i1 %exitcond, label %loop, label %exit, !llvm.loop !0
exit:
  ret void
}

!0 = !{!0, !{!"llvm.loop.unroll.disable", i1 0}}
