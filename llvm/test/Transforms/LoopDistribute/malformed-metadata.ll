; RUN: not opt -passes=loop-distribute -S < %s 2>&1 | FileCheck %s
;
; CHECK: llvm.loop metadata must have exactly two operands.

;define void @test(ptr nocapture %A, ptr nocapture readonly %B, i32 %Length) {
define void @test(i1 %exitcond) {
entry:
  br label %loop

loop:
  br i1 %exitcond, label %for.end.loopexit, label %loop, !llvm.loop !0

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}

!0 = !{!0, !{!"llvm.loop.distribute.enable"}}
