; RUN: opt -mtriple amdgcn-- -passes='print<uniformity>' -disable-output %s 2>&1 | FileCheck %s

; Test a call with "nodivergencesource" attribute and divergent argument.
; The result of the call should propagate divergence from the operand.

declare i32 @llvm.amdgcn.workitem.id.x()
declare i32 @nodivergencesourcefunc(i32) #0

; CHECK-LABEL: 'test_nodivergencesource_divergent_arg'
; CHECK: DIVERGENT: %tid
; CHECK: DIVERGENT: %call
define void @test_nodivergencesource_divergent_arg() {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %call = call i32 @nodivergencesourcefunc(i32 %tid)
  ret void
}

attributes #0 = { nounwind nodivergencesource }
