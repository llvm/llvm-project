; RUN: opt -mtriple=amdgpu-- -passes='print<uniformity>' -disable-output %s 2>&1 | FileCheck %s

; CHECK: DIVERGENT: %tid
; CHECK: DIVERGENT: %divergentval
; CHECK-NOT: DIVERGENT: %uniformval
; CHECK: %uniformval
define void @test() {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %divergentval = call i32 @normalfunc(i32 %tid)
  %uniformval = call i32 @alwaysuniformfunc(i32 %tid)
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() nounwind
declare i32 @normalfunc() nounwind
declare i32 @alwaysuniformfunc() alwaysuniform nounwind
