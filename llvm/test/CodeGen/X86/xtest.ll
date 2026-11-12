; RUN: llc < %s -mtriple=x86_64-- -mattr=+rtm | FileCheck %s

declare i32 @llvm.x86.xtest() nounwind

define i32 @test_xtest() nounwind uwtable {
entry:
  %0 = tail call i32 @llvm.x86.xtest() nounwind
  ret i32 %0
; CHECK: test_xtest
; CHECK: xtest
}
