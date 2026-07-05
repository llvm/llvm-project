; RUN: opt -mtriple amdgcn-- -passes='print<uniformity>' -disable-output %s 2>&1 | FileCheck %s

; CHECK-LABEL: 'test'
; CHECK: DIVERGENT: %divergentval
; CHECK-NOT: DIVERGENT: %uniformval
; CHECK: %uniformval
define void @test() {
  %divergentval = call i32 @normalfunc()
  %uniformval = call i32 @nodivergencesourcefunc()
  ret void
}

; Test a call with "nodivergencesource" attribute and divergent argument.
; The result of the call should propagate divergence from the operand.

; CHECK-LABEL: 'test_nodivergencesource_divergent_arg'
; CHECK: DIVERGENT: %tid
; CHECK: DIVERGENT: %call
define void @test_nodivergencesource_divergent_arg() {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %call = call i32 @nodivergencesourcefunc_arg(i32 %tid)
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
declare i32 @normalfunc() #0
declare i32 @nodivergencesourcefunc() #1
declare i32 @nodivergencesourcefunc_arg(i32) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind nodivergencesource }
