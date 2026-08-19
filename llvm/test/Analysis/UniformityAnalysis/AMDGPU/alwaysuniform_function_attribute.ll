; RUN: opt -mtriple=amdgpu-- -passes='print<uniformity>' -disable-output %s 2>&1 | FileCheck %s

; CHECK: DIVERGENT: %divergentval
; CHECK-NOT: DIVERGENT: %uniformval
; CHECK: %uniformval
define void @test() {
  %divergentval = call i32 @normalfunc()
  %uniformval = call i32 @alwaysuniformfunc()
  ret void
}

declare i32 @normalfunc() nounwind
declare i32 @alwaysuniformfunc() alwaysuniform nounwind
