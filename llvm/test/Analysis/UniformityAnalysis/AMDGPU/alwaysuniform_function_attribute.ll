; RUN: opt -mtriple=amdgpu-- -passes='print<uniformity>' -disable-output %s 2>&1 | FileCheck %s

; CHECK: DIVERGENT: %tid
; CHECK: DIVERGENT: %divergentval
; CHECK-NOT: DIVERGENT: %uniformval
; CHECK: %uniformval
define void @test_uniform_return_val() {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %divergentval = call i32 @normalfunc(i32 %tid)
  %uniformval = call i32 @alwaysuniformfunc(i32 %tid)
  ret void
}

; The alwaysuniform result is uniform per call, but because it is live across a
; loop that exits divergently, the value observed after the loop is temporally
; divergent.

; CHECK-LABEL: BLOCK entry
; CHECK: DIVERGENT: %tid
; CHECK-NOT: DIVERGENT: {{.*}}%uniformval.loop = call i32 @alwaysuniformfunc
; CHECK: %uniformloopval = call i32 @alwaysuniformfunc
; CHECK: DIVERGENT: %exitcond
; CHECK: DIVERGENT: %temporal = phi i32
define i32 @test_temporal_divergence() {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %uniformloopval = call i32 @alwaysuniformfunc(i32 %iv)
  %iv.next = add i32 %iv, 1
  %exitcond = icmp slt i32 %iv.next, %tid
  br i1 %exitcond, label %loop, label %exit

exit:
  %temporal = phi i32 [ %uniformloopval, %loop ]
  ret i32 %temporal
}

declare i32 @llvm.amdgcn.workitem.id.x() nounwind
declare i32 @normalfunc(i32) nounwind
declare i32 @alwaysuniformfunc(i32) alwaysuniform nounwind
