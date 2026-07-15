; RUN: llc -mtriple=sparc < %s | FileCheck %s

;; Verify that g1 (the output of first asm) is properly understood to
;; be clobbered by the call instruction, and moved out of the way
;; before it. (NOTE: remember delay slot; mov executes before call)

; CHECK-LABEL: test1:
; CHECK: ta       9
; CHECK: call dosomething
; CHECK: mov      %g1, %i0

define i32 @test1() nounwind {
entry:
  %0 = tail call i32 asm sideeffect "ta $1", "={r1},i"(i32 9) nounwind
  tail call void @dosomething() nounwind
  ret i32 %0
}

;; Also check using the value.
; CHECK-LABEL: test2:
; CHECK: ta       9
; CHECK: call dosomething
; CHECK: mov      %g1, %i0
; CHECK: mov      %i0, %g1
; CHECK: ta       10

define void @test2() local_unnamed_addr nounwind {
entry:
  %0 = tail call i32 asm sideeffect "ta $1", "={r1},i"(i32 9) nounwind
  tail call void @dosomething() nounwind
  tail call void asm sideeffect "ta $0", "i,{r1}"(i32 10, i32 %0) nounwind
  ret void
}

declare void @dosomething() local_unnamed_addr nounwind
