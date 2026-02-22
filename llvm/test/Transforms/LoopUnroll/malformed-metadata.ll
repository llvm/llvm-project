; RUN: not opt -passes=verify -S < %s 2>&1 | FileCheck %s
;
; CHECK: llvm.loop attribute must have exactly one operand

define void @test(ptr nocapture %A, i32 %N) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %i.next = add i32 %i, 1
  %cond = icmp slt i32 %i.next, %N
  br i1 %cond, label %loop, label %exit, !llvm.loop !0
exit:
  ret void
}

!0 = !{!0, !{!"llvm.loop.unroll.disable", i1 0}}
