; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder < %s

define i32 @test(i8** swifterror)
; CHECK: define i32 @test(i8** swifterror)
{
  ret i32 0
}
